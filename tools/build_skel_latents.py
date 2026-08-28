# -*- coding: utf-8 -*-
"""
build_skel_latents.py — 为 mid_clean 构建全新的 skel 数据 (旧 final_skeleton* 弃用).

三阶段流水线 (单脚本, 可断点续跑):
  1. from GT 图 (白底黑字, final_imgs_256/<id>.png) 提取 1px 骨架 (skimage skeletonize)
  2. 1px → 3px 膨胀 (scipy binary_dilation, 8-邻域, iterations=3), 存 PNG
  3. 3px skel 图 (白底黑线, 与 GT 书法图同极性) VAE encode → latent shard (与
     final_latents_mid_clean 同构: latents float16(N,4,32,32) + img_ids int64)

输出目录 (全部新命名, 不碰旧目录):
  final_skel1/                    # 1px 白底黑线 PNG
  final_skel3/                    # 3px 白底黑线 PNG (模型/VAE 用)
  final_skel_latents_mid_clean/   # VAE-encoded skel latent shards (ControlNet 输入)

用法 (远程后台):
  nohup /opt/conda/bin/python tools/build_skel_latents.py \
      --csv 5script/train_mid_clean.csv \
      --img-root final_imgs_256 \
      --skel1-dir final_skel1 --skel3-dir final_skel3 \
      --latent-out final_skel_latents_mid_clean \
      --workers 32 > /tmp/build_skel_latents.log 2>&1 &
"""
import os
import sys
import csv
import re
import time
import glob
import argparse

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 骨架提取 (带 scipy fallback, 与 eval 口径一致)
# ---------------------------------------------------------------------------
def _skel_impl():
    try:
        from skimage.morphology import skeletonize
        return skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure

        def skeletonize(binary):
            skel = np.zeros_like(binary)
            img = binary.copy()
            struct = generate_binary_structure(2, 2)
            while img.any():
                eroded = binary_erosion(img, structure=struct)
                skel |= img & ~eroded
                img = eroded
            return skel
        return skeletonize


_SKEL = None


def _dilate3(binary):
    from scipy.ndimage import binary_dilation, generate_binary_structure
    se = generate_binary_structure(2, 2)  # 8-邻域
    return binary_dilation(binary, structure=se, iterations=3)


def process_one(img_path, skel1_path, skel3_path):
    """读 GT 图 → 二值(笔画=暗) → 1px 骨架 → 3px 膨胀 → 存 白底黑线 PNG."""
    global _SKEL
    if _SKEL is None:
        _SKEL = _skel_impl()
    img = np.asarray(Image.open(img_path).convert("L"))
    binary = img < 127  # 白底黑字: 暗像素 = 笔画
    skel1 = _SKEL(binary)
    skel3 = _dilate3(skel1)
    # 存为 白底黑线 (线=0, 底=255): 与 GT 书法图同极性, VAE 空间对齐
    arr3 = np.where(skel3, 0, 255).astype(np.uint8)
    Image.fromarray(arr3, mode="L").save(skel3_path)
    arr1 = np.where(skel1, 0, 255).astype(np.uint8)
    Image.fromarray(arr1, mode="L").save(skel1_path)
    return os.path.basename(skel3_path)


# ---------------------------------------------------------------------------
# Phase 1/2: 多进程 提取+dilate 存 PNG (跳过已存在)
# ---------------------------------------------------------------------------
def build_pngs(csv_file, img_root, skel1_dir, skel3_dir, workers):
    os.makedirs(skel1_dir, exist_ok=True)
    os.makedirs(skel3_dir, exist_ok=True)
    ids = []
    with open(csv_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.png", row["image_path"])
            if m:
                ids.append(int(m.group(1)))
    ids = sorted(set(ids))
    print(f"[skel] {len(ids)} unique img ids from {csv_file}", flush=True)

    todo = []
    for iid in ids:
        sk3 = os.path.join(skel3_dir, f"{iid}.png")
        if os.path.exists(sk3):
            continue
        todo.append((os.path.join(img_root, f"{iid}.png"),
                     os.path.join(skel1_dir, f"{iid}.png"), sk3))
    print(f"[skel] todo: {len(todo)} / {len(ids)} (skip existing)", flush=True)

    import multiprocessing as mp
    t0 = time.time()
    done = 0
    with mp.Pool(workers) as pool:
        for _ in pool.starmap(process_one, todo, chunksize=64):
            done += 1
            if done % 20000 == 0 or done == len(todo):
                print(f"[skel] {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[skel] PNG phase DONE {done} in {time.time()-t0:.0f}s", flush=True)
    return ids


# ---------------------------------------------------------------------------
# Phase 3: VAE encode → latent shards (跳过已完成 shard)
# ---------------------------------------------------------------------------
def build_latents(ids, skel3_dir, latent_out, vae_path, shard_size=5000, scaling=0.18215):
    import torch
    from diffusers.models import AutoencoderKL

    os.makedirs(latent_out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(latent_out, "shard_*.npz")))
    done_ids = set()
    for sp in shards:
        with np.load(sp) as d:
            done_ids.update(int(x) for x in d["img_ids"])
    todo_ids = [i for i in ids if i not in done_ids]
    print(f"[vae] encode todo: {len(todo_ids)} / {len(ids)} (skip {len(done_ids)})", flush=True)
    if not todo_ids:
        print("[vae] nothing to encode", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    print(f"[vae] model loaded on {device}", flush=True)

    # 按 id 排序分批 encode, 每 shard_size 攒一个 shard
    import numpy as np
    batch = []
    t0 = time.time()
    n_done = 0

    def flush():
        nonlocal batch
        if not batch:
            return
        t_ids = [b[0] for b in batch]
        imgs = np.stack([b[1] for b in batch])          # (B,256,256) uint8 0/255
        x = (torch.from_numpy(imgs.astype(np.float32)) / 255.0 * 2.0 - 1.0)  # [-1,1]
        x = x.unsqueeze(1).repeat(1, 3, 1, 1).to(device)  # (B,3,256,256) 白底黑线
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * scaling  # (B,4,32,32)
        lat = lat.float().cpu().numpy().astype(np.float16)
        for j in range(0, len(t_ids), shard_size):
            chunk_t = t_ids[j:j + shard_size]
            chunk_l = lat[j:j + shard_size]
            sp = os.path.join(latent_out, f"shard_{chunk_t[0]:05d}_{chunk_t[-1]:05d}.npz")
            # 区间命名天然支持断点续跑: 已存在 shard 的 img_ids 会被跳过
            np.savez(sp, latents=chunk_l, img_ids=np.asarray(chunk_t, dtype=np.int64))
        batch = []

    with torch.no_grad():
        for iid in todo_ids:
            a = np.asarray(Image.open(os.path.join(skel3_dir, f"{iid}.png")).convert("L"))
            batch.append((iid, a))
            n_done += 1
            if len(batch) >= 64:
                flush()
                if n_done % 20000 == 0 or n_done == len(todo_ids):
                    print(f"[vae] {n_done}/{len(todo_ids)} ({time.time()-t0:.0f}s)", flush=True)
    flush()
    print(f"[vae] DONE {n_done} in {time.time()-t0:.0f}s -> {latent_out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_mid_clean.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--skel1-dir", default="final_skel1")
    ap.add_argument("--skel3-dir", default="final_skel3")
    ap.add_argument("--latent-out", default="final_skel_latents_mid_clean")
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    t_all = time.time()
    ids = build_pngs(args.csv, args.img_root, args.skel1_dir, args.skel3_dir, args.workers)
    build_latents(ids, args.skel3_dir, args.latent_out, args.vae_path)
    print(f"[skel] ALL DONE in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()