#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_gt_skel_w7.py — GT 加粗骨架: 两阶段 (先落盘栅格, 再批量编码)。

配方:
    GT 图 -> 灰度 -> ink = g < 128 -> skeletonize(1px)
         -> binary_dilation((w-1)//2) -> [-1,1] 256x256
         -> VAE encode(mode) * scaling -> (4,32,32) fp16

为什么 7px (实测量):
    32x32 latent 上一根 3px 骨架只有 0.375 格 —— 亚像素。纯 MSE 下
    「输出空白(0.379)」比「原样输出 std(0.416)」更接近 gt, 监督信号接近不可分;
    且任何栅格算子都会把细脊摊断。7px 折算 0.88 格, 可分辨。

★ 两阶段 (2026-09-29 用户裁定):
    stage=rasters : 多进程 骨架化+膨胀 -> **所有宽度的栅格全部落盘** (PNG)
    stage=encode  : 只读 PNG -> 分块 -> GPU 批量编码 -> shards
  分开的理由: 膨胀是 CPU 密集 (旧版在主进程单核跑十几分钟), 编码是 GPU 密集;
  混在一起会让两边互相等。落盘后编码可反复重跑, 不必重做膨胀。

用法:
    python tools/build_gt_skel_w7.py --stage rasters --csv <csv> --widths 3,7
    python tools/build_gt_skel_w7.py --stage encode  --csv <csv> --widths 7
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

_INK_THR = 128


def _rasters_one(args):
    """子进程: GT 图 -> 1px 骨架 -> 各宽度栅格。返回 {w: uint8 (H,W)}"""
    img_id, img_path, widths = args
    from scipy.ndimage import binary_dilation
    try:
        g = np.asarray(Image.open(img_path).convert("L"))
    except Exception:                                          # noqa: BLE001
        return img_id, None
    ink = g < _INK_THR
    if not ink.any():
        return img_id, None
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(ink)
    except Exception:                                          # noqa: BLE001
        from scipy.ndimage import binary_erosion, generate_binary_structure
        st = generate_binary_structure(2, 2)
        sk = np.zeros_like(ink)
        cur = ink.copy()
        while cur.any():
            er = binary_erosion(cur, structure=st)
            sk |= cur & ~er
            cur = er
    if not sk.any():
        return img_id, None
    out = {1: sk.astype(np.uint8)}                             # 1px 也留 (clDice 用)
    for w in widths:
        it = max(0, (w - 1) // 2)
        d = binary_dilation(sk, iterations=it) if it > 0 else sk
        out[w] = d.astype(np.uint8)
    return img_id, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rasters", "encode"], default="rasters")
    ap.add_argument("--csv", default="assets/train_top10_style23.csv")
    ap.add_argument("--widths", default="7")
    ap.add_argument("--png-root", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--out-root", default="data/top10_style23/shards_gtskel")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64,
                    help="GPU 编码 micro-batch (256x256 上 128 会 OOM)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--shard-size", type=int, default=5120)
    ap.add_argument("--sf", type=float, default=0.18215)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    widths = [int(x) for x in a.widths.split(",") if x.strip()]
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.limit > 0:
        rows = rows[:a.limit]
    jobs = [(int(r["img_id"]), r["image_path"]) for r in rows
            if r.get("img_id") and r.get("image_path")]
    print(f"[1] stage={a.stage} {len(jobs)} 条 widths={widths} workers={a.workers}",
          flush=True)

    # ── stage 1: 栅格全部落盘 ────────────────────────────────────────────
    if a.stage == "rasters":
        for w in set(widths) | {1}:
            os.makedirs(f"{a.png_root}_w{w}", exist_ok=True)
        nok = 0
        with Pool(a.workers) as pool:
            it = pool.imap_unordered(
                _rasters_one, [(i, p, widths) for i, p in jobs], chunksize=32)
            for k, (iid, out) in enumerate(it):
                if out is None:
                    continue
                for w, arr in out.items():
                    Image.fromarray(np.where(arr > 0, 0, 255).astype(np.uint8)).save(
                        f"{a.png_root}_w{w}/{iid:06d}.png")
                nok += 1
                if (k + 1) % 10000 == 0:
                    print(f"    栅格 {k + 1}/{len(jobs)} (有效 {nok})", flush=True)
        print(f"[2] 栅格落盘完成 {nok}/{len(jobs)}", flush=True)
        for w in set(widths) | {1}:
            d = f"{a.png_root}_w{w}"
            print(f"    {d}: {len(os.listdir(d))} 张", flush=True)
        return

    # ── stage 2: 只读 PNG, 批量编码 ─────────────────────────────────────
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "data/pretrained/pretrained_models/sd-vae-ft-ema").to(a.device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    def to_tensor(u8):
        a2 = np.where(u8 > 0, 0.0, 1.0).astype(np.float32)      # ink=黑(0)
        t = torch.from_numpy(a2)[None, None].repeat(1, 3, 1, 1)
        t = F.interpolate(t, size=(256, 256), mode="bilinear",
                          align_corners=False)
        return t * 2.0 - 1.0

    @torch.no_grad()
    def encode(ts):
        out = []
        for i in range(0, len(ts), a.batch):
            x = torch.cat(ts[i:i + a.batch]).to(a.device)
            z = vae.encode(x).latent_dist.mode() * a.sf
            out.append(z.half().cpu().numpy())
            del x, z
        return np.concatenate(out, 0)

    for w in widths:
        src = f"{a.png_root}_w{w}"
        if not os.path.isdir(src):
            print(f"[!] {src} 不存在, 先跑 --stage rasters", flush=True)
            continue
        avail = [iid for iid, _ in jobs
                 if os.path.exists(os.path.join(src, f"{iid:06d}.png"))]
        out_dir = f"{a.out_root}_w{w}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"[3] w{w}: {len(avail)} 张 -> {out_dir}", flush=True)
        lat_parts, id_parts = [], []
        for s in range(0, len(avail), a.chunk):
            part = avail[s:s + a.chunk]
            ts = []
            for iid in part:
                u8 = (np.asarray(Image.open(os.path.join(src, f"{iid:06d}.png"))
                                 .convert("L")) < _INK_THR).astype(np.uint8)
                ts.append(to_tensor(u8))
            lat_parts.append(encode(ts))
            id_parts.extend(str(i) for i in part)
            del ts
            print(f"    w{w} {min(s + a.chunk, len(avail))}/{len(avail)}", flush=True)
        lat = np.concatenate(lat_parts, 0)
        ids = np.array(id_parts)
        ns = 0
        for s in range(0, len(lat), a.shard_size):
            sl = slice(s, min(s + a.shard_size, len(lat)))
            np.savez(os.path.join(out_dir, f"shard_{ns:05d}.npz"),
                     latents=lat[sl], img_ids=ids[sl])
            ns += 1
        print(f"[4] w{w}: {len(lat)} 条 -> {out_dir} ({ns} shards) "
              f"latent std={float(np.asarray(lat, np.float32).std()):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
