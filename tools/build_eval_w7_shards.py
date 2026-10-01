#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_eval_w7_shards.py — 为 Strict84 和 Seen20 评测集生成 7px (w7) 粗骨架的评测分片 (GT 骨架目标 + 标准字骨架条件)"""
import os, sys, glob, csv, re, time
import numpy as np
import torch as th
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

dev = th.device("cuda" if th.cuda.is_available() else "cpu")

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

from src.eval.in_mem_eval import _get_vae
import torchvision.transforms as T

# 7px 膨胀核
struct22 = generate_binary_structure(2, 2)


def make_w7_skel(img_path):
    """读灰度图 -> 提取骨架 -> 膨胀到 7px -> 返回白底(255)黑线(0) uint8 数组"""
    a = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    ink = a < 128
    if skeletonize is not None:
        sk = skeletonize(ink)
    else:
        sk = ink
    # iterations=3 对应 7px 线宽
    sk7 = binary_dilation(sk, struct22, iterations=3)
    out = np.full(sk7.shape, 255, dtype=np.uint8)
    out[sk7] = 0
    return out


def encode_images_to_shard(img_arrays, img_ids, out_dir, vae):
    """将 (N, 256, 256) 白底黑线图像数组编码成 VAE latents 并保存 shard_00000.npz"""
    os.makedirs(out_dir, exist_ok=True)
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    tensors = []
    for a in img_arrays:
        im_rgb = Image.fromarray(a).convert("RGB")
        tensors.append(tf(im_rgb))
    x = th.stack(tensors).to(dev)
    
    lats = []
    with th.no_grad():
        for i in range(0, len(x), 16):
            batch_x = x[i:i+16]
            lat = (vae.encode(batch_x).latent_dist.mode() * 0.18215).float()
            lats.append(lat.cpu().numpy().astype(np.float16))
    all_lats = np.concatenate(lats, axis=0)
    all_ids = np.array(img_ids, dtype=np.int64)
    
    out_f = os.path.join(out_dir, "shard_00000.npz")
    np.savez_compressed(out_f, latents=all_lats, img_ids=all_ids)
    print(f"✓ 已生成并保存: {out_f} (N={len(all_ids)}, shape={all_lats.shape}, mean={all_lats.mean():.4f})")


def main():
    print(f"[1] 载入 VAE 到 {dev}...")
    vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
    
    # ── 1. Strict84 GT 骨架 (w7 目标) ──
    strict_csv = "assets/eval_top10_strict_subset84.csv"
    rows_strict = list(csv.DictReader(open(strict_csv, encoding="utf-8")))
    print(f"\n[2] 处理 Strict84 GT 骨架 (w7, 84 样本)...")
    gt_imgs, gt_ids = [], []
    for r in rows_strict:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        p = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(ROOT, r["image_path"])
        gt_imgs.append(make_w7_skel(p))
        gt_ids.append(iid)
    encode_images_to_shard(gt_imgs, gt_ids, "data/top10_style23/gt_skel_eval_strict84_w7", vae)
    
    # ── 2. Strict84 标准字骨架 (w7 条件) ──
    print(f"\n[3] 处理 Strict84 标准字骨架 (w7, 84 样本)...")
    std_imgs, std_ids = [], []
    for r in rows_strict:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        p = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join(ROOT, r["std_path"])
        std_imgs.append(make_w7_skel(p))
        std_ids.append(iid)
    encode_images_to_shard(std_imgs, std_ids, "data/top10_style23/std_eval_strict84_w7", vae)
    
    # ── 3. Seen20 标准字骨架 (w7 条件) ──
    seen_csv = "assets/eval_top10_seen_20.csv"
    rows_seen = list(csv.DictReader(open(seen_csv, encoding="utf-8")))
    print(f"\n[4] 处理 Seen20 标准字骨架 (w7, 20 样本)...")
    seen_std_imgs, seen_std_ids = [], []
    for r in rows_seen:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        p = r["std_path"] if os.path.isabs(r["std_path"]) else os.path.join(ROOT, r["std_path"])
        seen_std_imgs.append(make_w7_skel(p))
        seen_std_ids.append(iid)
    encode_images_to_shard(seen_std_imgs, seen_std_ids, "data/top10_style23/std_eval_seen20_w7", vae)
    
    print("\n🎉 评测集全部 w7 分片制作完毕，数据链路完美对齐！")


if __name__ == "__main__":
    main()
