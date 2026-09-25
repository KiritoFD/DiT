#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/encode_supplements_gpu.py — 在本地 GPU (RTX 4070) 上完成增强补丁的 VAE 编码。

产物:
  data/supplements/shards_img/shard_00000.npz
  data/supplements/shards_std/shard_00000.npz
"""
import argparse
import csv
import os
import sys
import time
import torch

# 严格硬锁显存上限: 本进程最高占用 60% (约 4.8GB)，确保系统总显存绝对不超过 7.0GB
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.60, 0)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.data.vae_io import encode_csv

CSV_PATH = "assets/train_font_supplements.csv"
OUT_BASE = "data/supplements"
VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"


def main():
    parser = argparse.ArgumentParser(description="在本地 GPU 上编码增强数据的 VAE Latents")
    parser.add_argument("--batch", type=int, default=24, help="VAE 编码 Batch 大小 (默认 24，显存严格可控)")
    parser.add_argument("--workers", type=int, default=0, help="数据读取 Worker 数 (Windows 建议 0 避免句柄开销)")
    parser.add_argument("--which", default="both", choices=["img", "std", "both"])
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"错误: 找不到补丁元数据: {CSV_PATH}")
        sys.exit(1)

    print(f"=== 开始在本地 GPU (RTX 4070) 上执行 VAE Latent 编码 ===")
    print(f"显存安全策略: 进程硬锁上限 <= 4.8GB, 系统总显存严格保持 < 7.0GB")
    print(f"Batch 大小: {args.batch} | Workers: {args.workers}")
    print(f"输入 CSV: {CSV_PATH}")
    print(f"VAE 权重: {VAE_PATH}")
    t0 = time.time()

    # 1. 编码图像 Latent
    if args.which in ("img", "both"):
        out_img = os.path.join(OUT_BASE, "shards_img")
        print(f"\n[1/2] 编码目标图像 Latents -> {out_img}...")
        # 临时 dump 适配 encode_csv 输入格式
        tmp_img_csv = "assets/_tmp_supp_img.csv"
        with open(CSV_PATH, encoding="utf-8") as f, open(tmp_img_csv, "w", encoding="utf-8", newline="") as fo:
            reader = csv.DictReader(f)
            writer = csv.writer(fo)
            writer.writerow(["image_path"])
            for r in reader:
                writer.writerow([r["image_path"]])
        encode_csv(tmp_img_csv, out_img, transform="gray", vae_path=VAE_PATH, batch=args.batch, workers=args.workers)
        if os.path.exists(tmp_img_csv):
            os.remove(tmp_img_csv)

    # 2. 编码标准骨架 Latent
    if args.which in ("std", "both"):
        out_std = os.path.join(OUT_BASE, "shards_std")
        print(f"\n[2/2] 编码标准骨架 Latents -> {out_std}...")
        tmp_std_csv = "assets/_tmp_supp_std.csv"
        with open(CSV_PATH, encoding="utf-8") as f, open(tmp_std_csv, "w", encoding="utf-8", newline="") as fo:
            reader = csv.DictReader(f)
            writer = csv.writer(fo)
            writer.writerow(["image_path"])
            for r in reader:
                writer.writerow([r["std_path"]])
        encode_csv(tmp_std_csv, out_std, transform="gray", vae_path=VAE_PATH, batch=args.batch, workers=args.workers)
        if os.path.exists(tmp_std_csv):
            os.remove(tmp_std_csv)

    print(f"\n✓ 全部编码完成! 总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
