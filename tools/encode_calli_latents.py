#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""encode_calli_latents.py — 用 Calli-VAE 正确约定批量生成 DiT latent shards。

与 tools/encode_aug_latents.py (sd-vae 用) 的唯一区别在 **编码刻度**:

    sd-vae :  z = vae.encode(x).latent_dist.mode() * 0.18215   (std≈1.17)
    Calli  :  z = vae.encode(x).latent_dist.sample()           (std≈0.97)  ← 本脚本

原因 (2026-10-09 事故):
    阶段一 Calli-VAE 的训练约定是 decode(posterior.sample() / 0.18215)。
    它的 posterior 方差大 (std≈0.88), decoder 必须带随机分量才能重建:
        decode(sample / 0.18215) ≈ 0.028  ✅
        decode(mode   / 0.18215) ≈ 0.608  ❌ 灰
    旧的 encode_aug_latents.py 用 mode*0.18215 -> v71 eval 全灰。
    ⇒ Calli latent 必须用 sample() 编码 (见 src/utils/calli_vae.py)。

decode 端不用改: 现有管线 `vae.decode(lat / 0.18215)` 对 (sample, /sf) 恰好自洽。
DiT 配置 vae_scaling_factor 仍填 0.18215。

产物 (每 shard): shard_%05d.npz
    latents : (N, 4, 32, 32) float16   (vae.encode(x).latent_dist.sample())
    img_ids : (N,) int64
    names   : (N,) unicode

用法:
    python tools/encode_calli_latents.py \
        --csv exp-std/csv/train_top10_aug_sym.csv \
        --out exp-std/data/shards_img_aug_calli \
        --vae experiments/calli_vae_dino/calli_vae_step_25000 \
        --img-root . --batch 64 --shard-size 5120 --workers 8 --seed 0

    # 自检 (编码前打印 encode->decode L1, 期望 ≈0.03)
    python tools/encode_calli_latents.py --csv ... --out ... --check

断点续编: 已存在的完整 shard 直接跳过 (按 shard 数 * shard-size 推算已编行数)。
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 注意: 这里 **不乘** scaling_factor (与 sd-vae 版不同)
SCALING_FACTOR_DECODE = 0.18215   # 仅用于自检 decode(z / sf)


class _AugImageDataset(torch.utils.data.Dataset):
    """按 CSV 行序读取原图 -> [-1,1] RGB (3,H,W)。"""

    def __init__(self, rows, img_root, image_size=256):
        self.rows = rows
        self.img_root = img_root
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p = r["image_path"]
        full = p if os.path.isabs(p) else os.path.join(self.img_root, p)
        name = os.path.basename(p)
        try:
            with Image.open(full) as im:
                img = im.convert("RGB")
                if img.size != (self.image_size, self.image_size):
                    img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            return t, str(r["img_id"]), name, True
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size), str(r["img_id"]), name, False


def _shard_path(out_dir, k):
    return os.path.join(out_dir, f"shard_{k:05d}.npz")


def _count_done_rows(out_dir, shard_size):
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    if not shards:
        return 0, 0
    last = shards[-1]
    try:
        with np.load(last) as d:
            n_last = len(d["img_ids"])
    except Exception:
        n_last = 0
    if n_last == shard_size:
        return len(shards), len(shards) * shard_size
    print(f"[resume] 末 shard 不完整 ({n_last} 行) -> 删除重编: {last}", flush=True)
    os.remove(last)
    return len(shards) - 1, (len(shards) - 1) * shard_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae", required=True, help="Calli-VAE 目录")
    ap.add_argument("--img-root", default=".")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=5120)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="posterior.sample 的随机种子 (保证可复现)")
    ap.add_argument("--check", action="store_true", help="只自检: encode->decode L1, 不写 shard")
    ap.add_argument("--encode-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    a = ap.parse_args()

    from diffusers.models import AutoencoderKL

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.encode_dtype]
    print(f"[encode] loading VAE from {a.vae} ...", flush=True)
    vae = AutoencoderKL.from_pretrained(a.vae).to(a.device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    sf = float(getattr(vae.config, "scaling_factor", SCALING_FACTOR_DECODE) or SCALING_FACTOR_DECODE)

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    total = len(rows)
    if "img_id" not in rows[0]:
        raise ValueError(f"{a.csv} 缺 img_id 列")
    print("=" * 78)
    print(f"[encode] csv={a.csv} rows={total:,}  out={a.out}  vae={a.vae}")
    print(f"[encode] convention: z = vae.encode(x).latent_dist.sample()   (sf={sf} 仅 decode 用)")
    print("=" * 78)

    gen = None
    if a.device.startswith("cuda"):
        gen = torch.Generator(device=a.device)
        gen.manual_seed(int(a.seed))

    # ---------------- 自检 ----------------
    if a.check:
        ds = _AugImageDataset(rows[: a.batch], a.img_root, a.image_size)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
        imgs, _, _, _ = next(iter(dl))
        imgs = imgs.to(a.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32 and a.device.startswith("cuda"))):
            dist = vae.encode(imgs).latent_dist
            z = dist.sample(gen) if gen is not None else dist.sample()
        zf = z.float()
        with torch.no_grad():
            dec = vae.decode((zf / sf)).sample.float()
        l1 = (dec - imgs.float()).abs().mean().item()
        print(f"[check] encode->decode L1 = {l1:.4f}   (latent std={zf.std().item():.4f}, 期望 L1≈0.03 / std≈0.97)")
        print("[check]", "PASS ✅" if l1 < 0.10 else "FAIL ❌ (约定不对, 应使用 sample 而非 mode)")
        return

    os.makedirs(a.out, exist_ok=True)

    n_done_shards, n_done = _count_done_rows(a.out, a.shard_size)
    if n_done >= total:
        print(f"[encode] 已完成 ({n_done:,} >= {total:,}), 无需编码")
        return

    ds = _AugImageDataset(rows[n_done:], a.img_root, a.image_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False,
                                     num_workers=a.workers, pin_memory=True, drop_last=False)

    shard_idx = n_done_shards
    lat_buf, id_buf, name_buf = [], [], []
    n_new = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, lat_buf, id_buf, name_buf
        if not lat_buf:
            return
        np.savez(_shard_path(a.out, shard_idx),
                 latents=np.stack(lat_buf).astype(np.float16),
                 img_ids=np.asarray(id_buf, dtype=np.int64),
                 names=np.asarray(name_buf))
        shard_idx += 1
        lat_buf, id_buf, name_buf = [], [], []

    with torch.no_grad():
        for imgs, ids, names, ok in dl:
            imgs = imgs.to(a.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32 and a.device.startswith("cuda"))):
                # ★ 正确约定: sample (不乘 scaling_factor)
                z = vae.encode(imgs).latent_dist.sample(gen) if gen is not None else vae.encode(imgs).latent_dist.sample()
            z = z.float().cpu().numpy()
            for k in range(z.shape[0]):
                if not bool(ok[k]):
                    continue
                lat_buf.append(z[k])
                id_buf.append(int(ids[k]))
                name_buf.append(str(names[k]))
                n_new += 1
                if len(lat_buf) >= a.shard_size:
                    flush()
            if n_new and n_new % (a.batch * 20) < a.batch:
                rate = n_new / max(1e-6, time.time() - t0)
                eta = (total - n_done - n_new) / max(1e-6, rate)
                print(f"[encode] {n_done + n_new:,}/{total:,} ({rate:.0f} 图/s, ETA {eta/60:.1f} min)", flush=True)
    flush()

    ids_all, names_all, n_rows = [], [], 0
    for sp in sorted(glob.glob(os.path.join(a.out, "shard_*.npz"))):
        with np.load(sp) as d:
            assert "names" in d, f"{sp} 缺 names 键"
            ids_all.append(d["img_ids"])
            names_all.append(d["names"])
            n_rows += len(d["img_ids"])
    ids_all = np.concatenate(ids_all)
    assert len(np.unique(ids_all)) == len(ids_all), "img_id 有重复"
    print("=" * 78)
    print(f"[encode] 完成: {n_rows:,} 行 / {len(glob.glob(os.path.join(a.out, 'shard_*.npz')))} shards "
          f"/ 耗时 {time.time() - t0:.0f}s")
    print(f"[encode] img_id 范围 [{ids_all.min():,}, {ids_all.max():,}] names 样例 {list(names_all[:2])}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
