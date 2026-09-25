# -*- coding: utf-8 -*-
"""
build_skel_latents.py — 为 mid_clean 构建全新的 skel 数据 (旧 data/skel/final_skeleton* 弃用).

三阶段流水线 (单脚本, 可断点续跑):
  1. from GT 图 (白底黑字, data/imgs/final_imgs_256/<id>.png) 提取 1px 骨架 (skimage skeletonize)
  2. 1px → 3px 膨胀 (scipy binary_dilation, 8-邻域, iterations=3), 存 PNG
  3. 3px skel 图 (白底黑线, 与 GT 书法图同极性) VAE encode → latent shard (与
     data/latents/final_latents_mid_clean 同构: latents float16(N,4,32,32) + img_ids int64)

输出目录 (全部新命名, 不碰旧目录):
  data/skel/final_skel1/                    # 1px 白底黑线 PNG
  data/skel/final_skel3/                    # 3px 白底黑线 PNG (模型/VAE 用)
  data/skel/final_skel_latents_mid_clean/   # VAE-encoded skel latent shards (ControlNet 输入)

用法 (远程后台):
  nohup /opt/conda/bin/python tools/build_skel_latents.py \
      --csv assets/train_mid_clean.csv \
      --img-root data/imgs/final_imgs_256 \
      --skel1-dir data/skel/final_skel1 --skel3-dir data/skel/final_skel3 \
      --latent-out data/skel/final_skel_latents_mid_clean \
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


def _thicken(binary, width):
    """1px 骨架 -> **精确** width 像素宽的粗骨架。

    为什么不用 _dilate3 的迭代膨胀: 8-邻域膨胀 n 次得到的宽度是 ~2n+1 且随
    局部走向变化 (斜线比直线粗), 无法精确对准目标宽度。EDT 半径法给出的是
    欧氏意义上的等宽笔画带, 与目标宽度严格一致, 且只需一次变换 (更快)。

    用法: distance_transform_edt(~binary) 给每个**背景**像素到最近骨架像素的
    欧氏距离; 保留 d <= (width-1)/2 即得总宽 ≈ width 的带。
    """
    if width <= 1:
        return binary
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(~binary)
    return d <= (float(width) - 1.0) / 2.0


def process_one(img_path, skel1_path, skel3_path, width=3):
    """读 GT 图 → 二值(笔画=暗) → 1px 骨架 → width px 加粗 → 存 白底黑线 PNG."""
    global _SKEL
    if _SKEL is None:
        _SKEL = _skel_impl()
    img = np.asarray(Image.open(img_path).convert("L"))
    binary = img < 127  # 白底黑字: 暗像素 = 笔画
    skel1 = _SKEL(binary)
    skel3 = _thicken(skel1, width)
    # 存为 白底黑线 (线=0, 底=255): 与 GT 书法图同极性, VAE 空间对齐
    arr3 = np.where(skel3, 0, 255).astype(np.uint8)
    Image.fromarray(arr3, mode="L").save(skel3_path)
    arr1 = np.where(skel1, 0, 255).astype(np.uint8)
    Image.fromarray(arr1, mode="L").save(skel1_path)
    return os.path.basename(skel3_path)


# ---------------------------------------------------------------------------
# Phase 1/2: 多进程 提取+dilate 存 PNG (跳过已存在)
# ---------------------------------------------------------------------------
def build_pngs(csv_file, img_root, skel1_dir, skel3_dir, workers, width=3,
               id_width=0):
    """id_width: 文件名补零位数。0=不补零(历史 fame 数据集), 6=000000.png(50k 数据集)。

    ⚠ 这个坑踩过: 50k 的 image_path 是 `data/50k/imgs/000000.png` (6 位补零),
    而这里原本拼 f"{iid}.png" -> "0.png" 打不开 -> 静默全零/FileNotFound。
    """
    def _name(iid):
        return f"{iid:0{id_width}d}.png" if id_width > 0 else f"{iid}.png"

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
        sk3 = os.path.join(skel3_dir, _name(iid))
        if os.path.exists(sk3):
            continue
        todo.append((os.path.join(img_root, _name(iid)),
                     os.path.join(skel1_dir, _name(iid)), sk3, width))
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
def build_latents(ids, skel3_dir, latent_out, vae_path, shard_size=5000, scaling=0.18215,
                  id_width=0):
    def _name(iid):
        return f"{iid:0{id_width}d}.png" if id_width > 0 else f"{iid}.png"
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
            a = np.asarray(Image.open(os.path.join(skel3_dir, _name(iid))).convert("L"))
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
    ap.add_argument("--csv", default="assets/train_mid_clean.csv")
    ap.add_argument("--img-root", default="data/imgs/final_imgs_256")
    ap.add_argument("--skel1-dir", default="data/skel/final_skel1")
    ap.add_argument("--skel3-dir", default="data/skel/final_skel3")
    ap.add_argument("--latent-out", default="data/skel/final_skel_latents_mid_clean")
    ap.add_argument("--vae-path",
                    default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--latent-src", default="3px", choices=["1px", "3px"],
                    help="编码哪一套骨架进 latent。1px = 细骨架(1px PNG), "
                         "3px = 膨胀后(默认, 与历史 ControlNet 一致)。"
                         "PNG 阶段两套都会生成, 这里只选 latent 的输入源。")
    # ★ 骨架加粗宽度 (像素)。默认 3 = 历史行为 (迭代膨胀的近似宽度)。
    #   中程结构载体实测最优 = 20px (两条独立证据:
    #     ① probe_coarse_skel.py 的 latent 残差扫描在 20px 取到最小值;
    #     ② mid_skel_iou.py 统计 GT 真实笔画宽度中位数 = 19.4px)。
    ap.add_argument("--dilate-width", type=int, default=3, dest="dilate_width")
    ap.add_argument("--id-width", type=int, default=0, dest="id_width",
                    help="文件名补零位数: 6=000000.png (50k 数据集), 0=不补零 (fame)")
    args = ap.parse_args()

    t_all = time.time()
    ids = build_pngs(args.csv, args.img_root, args.skel1_dir, args.skel3_dir,
                     args.workers, args.dilate_width, args.id_width)
    src_dir = args.skel1_dir if args.latent_src == "1px" else args.skel3_dir
    print(f"[skel] latent-src = {args.latent_src} -> {src_dir}", flush=True)
    build_latents(ids, src_dir, args.latent_out, args.vae_path, id_width=args.id_width)
    print(f"[skel] ALL DONE in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()