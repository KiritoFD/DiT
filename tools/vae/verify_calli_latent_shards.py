#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_calli_latent_shards.py — 校验 Calli-VAE 预编码 latent shards 的"约定正确性"。

为什么必须单独跑 (2026-10-09 v71 事故):
  同一条"潜变量约定"出现在 三处 (编码 / 存盘 / 解码), 只要一处不一致, 推理端就出灰图:
    v71: 存裸 sample()(不乘sf) + 旧 VAE(只认 decode(sample/sf))  -> 能跑但依赖非标准解码器
    本次: 存 sample()*sf     + 新 VAE(标准 decode(sample))       -> 标准约定, 无需补丁
  本工具用"四种解码输入的 L1 对照"把约定从口头约定变成实测契约, 并把结果写进 PROVENANCE.json。

检查项:
  1. shard 齐备 / names 键齐全 (数据集会逐行硬校验) / img_id 唯一
  2. 存储 latent 的逐维 std ≈ scaling_factor  (证明存的是 sample*sf, 不是裸 sample)
  3. decode(lat/sf) 的 L1 ≈ 0.01 (解析路径正确); 对照 decode(lat) 必须显著更差
  4. 产出 PROVENANCE.json (ok 字段供启动脚本预检)

用法:
  PYTHONPATH=. python tools/vae/verify_calli_latent_shards.py \
      --shards exp-std/data/shards_img_aug_calli_kl1e6_s12500_rsample \
      --vae experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv \
      --csv exp-std/csv/train_top10_aug_sym.csv --img-root .
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from diffusers import AutoencoderKL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", default=".")
    ap.add_argument("--n-check", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write-provenance", action="store_true",
                    help="把实测结果写成 <shards>/PROVENANCE.json")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
    sf = float(vae.config.scaling_factor)
    shards = sorted(glob.glob(os.path.join(a.shards, "shard_*.npz")))
    print(f"[verify] {len(shards)} shards | vae={a.vae} | sf={sf}")
    assert shards, "没有 shard!"

    n_rows, ids, miss_names = 0, [], 0
    for sp in shards:
        with np.load(sp) as d:
            if "names" not in d:
                miss_names += 1
            n_rows += len(d["img_ids"])
            ids.append(d["img_ids"])
    ids = np.concatenate(ids)
    uniq = len(np.unique(ids)) == len(ids)
    print(f"[verify] 行数 {n_rows:,} | 缺 names 的 shard {miss_names} | img_id 唯一 {uniq}")

    with np.load(shards[0]) as d:
        lat = torch.from_numpy(d["latents"][: a.n_check]).float().to(dev)
        ids8 = d["img_ids"][: a.n_check]
    path_by_id = {}
    with open(a.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path_by_id[int(r["img_id"])] = r["image_path"]
    imgs, used = [], []
    for i in ids8:
        rel = path_by_id.get(int(i))
        if rel is None:
            continue
        p = rel if os.path.isabs(rel) else os.path.join(a.img_root, rel)
        im = Image.open(p).convert("RGB").resize((256, 256))
        arr = np.asarray(im, np.float32) / 127.5 - 1.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        used.append((int(i), rel))
    assert imgs, "取不到参考图"
    lat = lat[: len(imgs)]
    x = torch.stack(imgs).to(dev)
    print(f"[verify] 自检样本: {used[:3]} ... 共 {len(imgs)}")

    with torch.no_grad():
        d_std = vae.decode(lat / sf).sample.clamp(-1, 1)
        d_raw = vae.decode(lat).sample.clamp(-1, 1)
        l1_std = F.l1_loss(d_std, x).item()
        l1_raw = F.l1_loss(d_raw, x).item()
    std_raw = lat.std().item()
    print(f"[verify] 存储 latent std = {std_raw:.4f} (期望 ≈ sf={sf:.4f} -> 说明存的是 sample*sf)")
    print(f"[verify] decode(lat/sf) L1 = {l1_std:.4f} | 对照 decode(lat) L1 = {l1_raw:.4f}")
    ok = bool(l1_std < 0.05 and l1_raw > 3 * l1_std and miss_names == 0 and uniq
              and abs(std_raw - sf) < 0.05)
    print(f"[verify] 总体 {'PASS ✅' if ok else 'FAIL ❌'}")

    if a.write_provenance:
        prov = dict(artifact="DiT latent shards", shards_dir=a.shards, n_shards=len(shards),
                    n_rows=int(n_rows), vae_dir=a.vae, scaling_factor=sf,
                    encode_convention="z_stored = posterior.sample() * scaling_factor",
                    decode_convention="x = vae.decode(z_stored / scaling_factor)",
                    latent_std_measured=round(std_raw, 4), selfcheck_decode_l1=round(l1_std, 4),
                    selfcheck_contrast=dict(decode_lat=round(l1_raw, 4)),
                    source_csv=a.csv, created="2026-10-09", ok=ok)
        out = os.path.join(a.shards, "PROVENANCE.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(prov, f, ensure_ascii=False, indent=2)
        print(f"[verify] 写入 {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
