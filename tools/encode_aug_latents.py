#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""encode_aug_latents.py — 为"对称笔画粗细增强"数据集批量生成 VAE latent shards。

与既有 tools/build_std_skel_widths.py / encode_top10_aug.py 的区别 (三条都是硬需求):
  1. **路径全部参数化** (原 encode_top10_aug.py 把 /home/ds/... 写死在源码里, 换机器就废)。
  2. **额外写 `names` 键** = 该行 image_path 的 basename。
     理由: MCCDLatentDataset 查表**只认 img_id**, 若小号段指向大库会静默取到别的字的
     latent (见 src/utils/latent_dataset.py `_check_shard_names` 的注释, 2026-09-21
     few-shot 就这么废掉过一整轮实验)。写了 names 后数据集会逐行硬校验, 错位直接抛错。
     ⚠ _load_shard_names 是"任一 shard 缺 names 就整体放弃校验", 所以必须每个 shard 都写。
  3. **img_id 取 CSV 的显式列**, 不用文件名正则。
     增强行的 image_path 形如 `0006533_tp.png`, 正则 `(\d+)\.png` 匹配不到 (".png" 前是 "tp"),
     原脚本会退化成 -1 → 全表覆盖同一个 id。

产物 (每 shard): shard_%05d.npz
  latents : (N, 4, 32, 32) float16    (vae.encode(x).latent_dist.mode()|sample() * 0.18215,
                                       由 --encode-mode 决定, 默认 mode)
  img_ids : (N,) int64
  names   : (N,) unicode              (image_path 的 basename)

用法 (远端 4090):
  python tools/encode_aug_latents.py \
      --csv exp-std/csv/train_top10_aug_sym.csv \
      --out exp-std/data/shards_img_aug \
      --vae pretrained_models/sd-vae-ft-ema \
      --img-root . --batch 256 --shard-size 5120 --workers 8

断点续编: 已存在的完整 shard 直接跳过 (按 shard 数 * shard-size 推算已编行数)。
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCALING_FACTOR = 0.18215


class _AugImageDataset(Dataset):
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
    """返回 (完整 shard 数, 已编行数)。最后一个不完整的 shard 会被删除重编。"""
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
    ap.add_argument("--vae", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--img-root", default=".", help="相对 image_path 的基准目录")
    ap.add_argument("--batch", type=int, default=64,
                    help="默认 64: batch 256 在 24G 卡上会 OOM —— VAE 的 group_norm 被 "
                         "autocast 强制上采 fp32, 单次要分配 8.6GB (实测 4090 直接 "
                         "CUDA OOM: Tried to allocate 8.00 GiB)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=5120)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-mode", default="mode", choices=["mode", "sample"],
                    help="潜空间编码约定。'mode' = latent_dist.mode()*sf (sd-vae-ft-ema 适用, "
                         "其后验 std≈2e-4, 与 sample 等价)。'sample' = latent_dist.sample()*sf "
                         "(**Calli-VAE 必须用这个**: 其后验极宽 std≈0.88, mode 无法承载细节, "
                         "用 mode 编码会与训练/解码约定不一致 -> 灰图)。")
    # ★ 2026-10-10: sf 不再硬编码。
    #   教训(E5): 原实现写死 SCALING_FACTOR=0.18215(= SD 官方拟合值), 换 Calli-VAE 后仍照用 ——
    #   而我们 VAE 的 raw latent 本来就是 N(0,1) 级(实测 mean=-0.115/std=1.120), 被它乘完只剩
    #   std 0.204 -> FM 目标里数据分量只有 1/5 -> 细节梯度被压小, 且辅助损失相对权重被放大 ~28x。
    #   规约见 docs/vae_latent_contract.md I2: **sf 与 shift 都必须由实测拟合**。
    ap.add_argument("--scaling-factor", default="auto",
                    help="'auto' = 用 --calib-n 张图实测拟合 (sf = 1/std(z_raw), shift = mean(z_raw)); "
                         "'none' = 不缩放不去均值 (sf=1, shift=0); 或给具体浮点值(如 0.18215, 仅复现旧数据用)")
    ap.add_argument("--calib-n", type=int, default=512,
                    help="--scaling-factor auto 时的标定样本数 (默认 512; std 估计误差 <1%)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    total = len(rows)
    print("=" * 78)
    print(f"[encode] csv={a.csv} rows={total:,}")
    print(f"[encode] out={a.out} vae={a.vae} batch={a.batch} shard_size={a.shard_size}")
    print("=" * 78)

    n_done_shards, n_done = _count_done_rows(a.out, a.shard_size)
    if n_done >= total:
        print(f"[encode] 已完成 ({n_done:,} >= {total:,}), 无需编码")
        return

    # 缺列早退 (img_id 是硬需求)
    if "img_id" not in rows[0]:
        raise ValueError(f"{a.csv} 缺 img_id 列, 拒绝用文件名正则兜底 (会错位)")

    ds = _AugImageDataset(rows[n_done:], a.img_root, a.image_size)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False,
                    num_workers=a.workers, pin_memory=True, drop_last=False)

    from diffusers.models import AutoencoderKL
    print(f"[encode] loading VAE from {a.vae} ...", flush=True)
    vae = AutoencoderKL.from_pretrained(a.vae).to(a.device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ★ 2026-10-10: sf / shift 由**实测拟合**（合同 docs/vae_latent_contract.md I2）。
    #   存盘约定: z_stored = (z_raw - SHIFT) * SF ; 解码 decode(z_stored/SF + SHIFT)。
    #   标定只在最前面的 --calib-n 张图上做一遍（std 估计误差 <1%），不额外占用主循环。
    import math
    _sfs = str(a.scaling_factor).lower()
    if _sfs == "auto":
        n_cal = int(min(a.calib_n, max(1, total - n_done)))
        # ★ 跨全表**等距抽样**, 不取前 N 行: 字帖的逐图 latent 均值随 字/书家/书体 变化,
        #   而 CSV 开头常是一批同书家样本 -> 取前 N 行会把 mean 估偏(实测 16 张估 -0.140 vs 全量 -0.115)。
        _pool = list(range(n_done, total))
        _stepc = max(1, len(_pool) // n_cal)
        cal_rows = [rows[j] for j in _pool[::_stepc][:n_cal]]
        ds_cal = _AugImageDataset(cal_rows, a.img_root, a.image_size)
        dl_cal = DataLoader(ds_cal, batch_size=a.batch, shuffle=False,
                            num_workers=min(int(a.workers), 4), pin_memory=True, drop_last=False)
        s1 = s2 = cnt = 0.0
        with torch.no_grad():
            for _imgs, _ids, _names, _ok in dl_cal:
                _imgs = _imgs.to(a.device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _d = vae.encode(_imgs).latent_dist
                    _lat = _d.sample() if a.encode_mode == "sample" else _d.mode()
                _x = _lat.float().double()
                s1 += float(_x.sum()); s2 += float((_x ** 2).sum()); cnt += float(_x.numel())
        _mu = s1 / max(cnt, 1.0)
        _var = max(s2 / max(cnt, 1.0) - _mu ** 2, 0.0)
        _std = math.sqrt(_var)
        SF, SHIFT = 1.0 / _std, _mu
        print(f"[encode] ★ auto 标定({n_cal} 张): raw mean={_mu:+.5f} std={_std:.5f} "
              f"-> SF={SF:.6f} SHIFT={SHIFT:+.5f}  (存盘应 ~N(0,1))", flush=True)
    elif _sfs == "none":
        SF, SHIFT = 1.0, 0.0
        print("[encode] scaling-factor=none -> SF=1 SHIFT=0 (存 raw latent)", flush=True)
    else:
        SF, SHIFT = float(a.scaling_factor), 0.0
        print(f"[encode] scaling-factor={SF} 显式指定, SHIFT=0 —— 仅用于复现旧数据", flush=True)
    print(f"[encode] ★ 调用方必须把 SF={SF!r} SHIFT={SHIFT!r} 写进 shards PROVENANCE 与 DiT config "
          f"(vae_scaling_factor / vae_shift_factor)", flush=True)

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
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _dist = vae.encode(imgs).latent_dist
                _lat = _dist.sample() if a.encode_mode == "sample" else _dist.mode()
                z = (_lat - SHIFT) * SF          # ★ 合同 I2: 去均值 + 归一化尺度
            z = z.float().cpu().numpy()
            for k in range(z.shape[0]):
                if not bool(ok[k]):
                    continue                      # 读图失败的行: 不写入 (数据集会报缺失)
                lat_buf.append(z[k])
                id_buf.append(int(ids[k]))
                name_buf.append(str(names[k]))
                n_new += 1
                if len(lat_buf) >= a.shard_size:
                    flush()
            if n_new and n_new % (a.batch * 20) < a.batch:
                rate = n_new / max(1e-6, time.time() - t0)
                eta = (total - n_done - n_new) / max(1e-6, rate)
                print(f"[encode] {n_done + n_new:,}/{total:,} "
                      f"({rate:.0f} 图/s, ETA {eta/60:.1f} min)", flush=True)
    flush()

    # 汇总校验
    ids_all, names_all, n_rows = [], [], 0
    for sp in sorted(glob.glob(os.path.join(a.out, "shard_*.npz"))):
        with np.load(sp) as d:
            assert "names" in d, f"{sp} 缺 names 键 -> 数据集会放弃内容校验"
            ids_all.append(d["img_ids"])
            names_all.append(d["names"])
            n_rows += len(d["img_ids"])
    ids_all = np.concatenate(ids_all)
    names_all = np.concatenate(names_all)
    assert len(np.unique(ids_all)) == len(ids_all), "img_id 有重复 -> 数据集查表会冲突"
    print("=" * 78)
    print(f"[encode] 完成: {n_rows:,} 行 / {len(glob.glob(os.path.join(a.out, 'shard_*.npz')))} shards "
          f"/ 耗时 {time.time() - t0:.0f}s")
    print(f"[encode] img_id 范围 [{ids_all.min():,}, {ids_all.max():,}] "
          f"names 样例 {list(names_all[:2])}")
    print(f"[encode] CSV 行数 {total:,} vs 落盘 {n_rows:,} "
          f"(差 {total - n_rows} = 读图失败行)")

    # ★ 2026-10-10: 编码器自己写 PROVENANCE（合同 §4 步骤 5 的硬要求）。
    #   过去 sf/shift 只存在于调用方脚本里, 换 VAE 时无人核对 -> E5。写在这里=与数据同生共死。
    _prov = {
        "artifact": "DiT latent shards",
        "shards_dir": a.out, "n_shards": len(glob.glob(os.path.join(a.out, "shard_*.npz"))),
        "n_rows": int(n_rows), "source_csv": a.csv, "img_root": a.img_root,
        "vae_dir": a.vae, "encode_mode": a.encode_mode,
        "encode_convention": ("z_stored = (posterior.%s() - SHIFT) * SF"
                              % ("sample" if a.encode_mode == "sample" else "mode")),
        "decode_convention": "x = vae.decode(z_stored / SF + SHIFT)",
        "scaling_factor": float(SF), "shift_factor": float(SHIFT),
        "scaling_factor_arg": str(a.scaling_factor), "calib_n": int(a.calib_n),
        "raw_mean_measured": (float(SHIFT) if _sfs == "auto" else None),
        "raw_std_measured": (float(1.0 / SF) if SF else None),
        "contract": "docs/vae_latent_contract.md I2: std(z_stored)~1 且 mean~0; SF/SHIFT 必须由实测拟合",
        "created": time.strftime("%F %T"),
    }
    with open(os.path.join(a.out, "PROVENANCE.json"), "w", encoding="utf-8") as f:
        json.dump(_prov, f, ensure_ascii=False, indent=2)
    print(f"[encode] ★ PROVENANCE -> {a.out}/PROVENANCE.json  "
          f"(SF={SF!r} SHIFT={SHIFT!r}) —— DiT config 必须照抄这两个数")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
