#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_std_skel_widths.py — 预建「多宽度」标准骨架 latent shards。

背景: 中程结构 loss 的 `dilate_skel` 载体需要不同笔宽的骨架 latent。与其在 latent 域
max-pool 近似膨胀 (structure_mid.latent_dilate), 不如在**像素域精确膨胀后各自 VAE 编码**
成一套按宽度分目录的 shards, 供 loss 按 k(t) 直接取用、也供校准脚本量 k*(t)。

流程 (每个目标宽度 w px):
  std 骨架 PNG (CSV 的 std_path) -> 二值化(ink<128) -> skeletonize 到 1px
  -> binary_dilation iterations=(w-1)//2 得到 ~w px 笔宽 -> [-1,1] 256x256
  -> VAE encode .latent_dist.mode() * scaling_factor -> (4,32,32) fp16
输出: <out_root>/shard_%05d.npz, 键 latents (N,4,32,32) fp16 + img_ids (N,) 字符串
      (与 shards_std 同键规则: 按 img_id 查表)

用法 (远端 GPU):
  python tools/build_std_skel_widths.py --csv assets/train_50k_v2.csv \
      --widths 3,5,7,9,11 --out-root data/50k/shards_std_w --batch 128
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_50k_v2.csv")
    ap.add_argument("--std-col", default="std_path")
    ap.add_argument("--widths", default="3,5,7,9,11")
    ap.add_argument("--out-root", default="data/50k/shards_std_w")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--sf", type=float, default=0.18215, help="vae_scaling_factor")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    widths = [int(x) for x in a.widths.split(",") if x.strip()]
    from scipy.ndimage import binary_dilation
    try:
        from skimage.morphology import skeletonize
        HAVE_SKEL = True
    except Exception:
        HAVE_SKEL = False
        print("[warn] skimage 不可用 -> 直接用二值图(不 skeletonize), 宽度会偏大")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("data/pretrained/sd-vae-ft-ema").to(a.device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    # img_id 与既有 shards 对齐: 优先显式列, 回退文件名
    def img_id(r):
        for c in ("img_id", "old_50k_id"):
            if str(r.get(c, "")).strip():
                return int(r[c])
        return int(os.path.splitext(os.path.basename(r["image_path"]))[0])

    # 预读所有 std 二值图 (一次), 各宽度复用
    items = []
    miss = 0
    for r in rows:
        sp = r.get(a.std_col, "")
        if not sp or not os.path.exists(sp):
            miss += 1
            continue
        g = np.asarray(Image.open(sp).convert("L"))
        ink = g < 128
        if HAVE_SKEL and ink.sum() > 0:
            ink = skeletonize(ink)
        items.append((img_id(r), ink))
    print(f"[read] {len(items)} 张 std 骨架 (缺 {miss}) | widths={widths} | skeletonize={HAVE_SKEL}")

    @torch.no_grad()
    def encode_batch(tensors):
        x = torch.stack(tensors).to(a.device)
        z = vae.encode(x).latent_dist.mode() * a.sf
        return z.half().cpu().numpy()

    def to_tensor(ink):
        a2 = np.where(ink, 0.0, 1.0).astype(np.float32)   # ink=黑(0), 背景=白(1)
        t = torch.from_numpy(a2)[None, None].repeat(3, 1, 1)
        t = F.interpolate(t[None], size=(256, 256), mode="bilinear", align_corners=False)[0]
        return t * 2.0 - 1.0                               # [-1,1]

    for w in widths:
        iters = max(0, (w - 1) // 2)
        out_dir = f"{a.out_root}_w{w}"
        os.makedirs(out_dir, exist_ok=True)
        lat_acc, id_acc, shard = [], [], 0

        def flush(force=False):
            nonlocal lat_acc, id_acc, shard
            if not lat_acc:
                return
            if not force and sum(x.shape[0] for x in lat_acc) < a.shard_size:
                return
            latents = np.concatenate(lat_acc, 0)
            img_ids = np.array(id_acc)
            np.savez(os.path.join(out_dir, f"shard_{shard:05d}.npz"),
                     latents=latents, img_ids=img_ids)
            shard += 1
            lat_acc, id_acc = [], []

        buf, bids = [], []
        for iid, ink in items:
            d = binary_dilation(ink, iterations=iters) if iters > 0 else ink
            buf.append(to_tensor(d)); bids.append(str(iid))
            if len(buf) >= a.batch:
                lat_acc.append(encode_batch(buf)); id_acc.extend(bids)
                buf, bids = [], []
                flush()
        if buf:
            lat_acc.append(encode_batch(buf)); id_acc.extend(bids)
        flush(force=True)
        n = sum(x.shape[0] for x in lat_acc) if lat_acc else 0
        print(f"[w{w}] 完成 -> {out_dir}  ({shard} shards)")


if __name__ == "__main__":
    main()
