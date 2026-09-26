#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/encode_glyph15k_gpu.py — 在本地 GPU (RTX 4070) 上对 15,548 张补齐样本进行 VAE 编码

产物:
  data/supplements_glyph/shards_img/shard_*.npz
  data/supplements_glyph/shards_std/shard_*.npz
"""
import argparse
import csv
import os
import sys
import time
import torch

# 显存安全保护：硬锁 75% (~6.0GB)，防止占用笔记本系统显存
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.data.vae_io import encode_csv

CSV_PATH = "assets/train_font_supplements_glyph15k.csv"
OUT_BASE = "data/supplements_glyph"
VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"


def main():
    parser = argparse.ArgumentParser(description="在本地 GPU 上编码 15.5k 补齐样本的 VAE Latents")
    parser.add_argument("--batch", type=int, default=16, help="VAE 编码 Batch 大小")
    parser.add_argument("--workers", type=int, default=0, help="数据读取 Worker 数")
    parser.add_argument("--which", default="both", choices=["img", "std", "both"])
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"错误: 找不到补丁元数据: {CSV_PATH}")
        sys.exit(1)

    print(f"=== 本地 GPU (RTX 4070) VAE Latent 编码器启动 ===")
    print(f"输入 CSV: {CSV_PATH}")
    print(f"输出根目录: {OUT_BASE}")
    print(f"VAE 模型: {VAE_PATH}")
    print(f"Batch Size: {args.batch} | Workers: {args.workers}")
    t0 = time.time()

    # 1. 编码目标图像 Latents
    if args.which in ("img", "both"):
        out_img = os.path.join(OUT_BASE, "shards_img")
        print(f"\n[1/2] 开始编码目标图像 Latents -> {out_img}...")
        tmp_img_csv = "assets/_tmp_glyph15k_img.csv"
        with open(CSV_PATH, encoding="utf-8") as f, open(tmp_img_csv, "w", encoding="utf-8", newline="") as fo:
            reader = csv.DictReader(f)
            writer = csv.writer(fo)
            writer.writerow(["image_path"])
            for r in reader:
                writer.writerow([r["image_path"]])
        encode_csv(tmp_img_csv, out_img, transform="gray", vae_path=VAE_PATH, batch=args.batch, workers=args.workers)
        if os.path.exists(tmp_img_csv):
            os.remove(tmp_img_csv)

    # 2. 编码标准骨架 Latents
    if args.which in ("std", "both"):
        out_std = os.path.join(OUT_BASE, "shards_std")
        print(f"\n[2/2] 开始编码标准骨架 Latents -> {out_std}...")
        tmp_std_csv = "assets/_tmp_glyph15k_std.csv"
        with open(CSV_PATH, encoding="utf-8") as f, open(tmp_std_csv, "w", encoding="utf-8", newline="") as fo:
            reader = csv.DictReader(f)
            writer = csv.writer(fo)
            writer.writerow(["image_path"])
            for r in reader:
                writer.writerow([r["std_path"]])
        encode_csv(tmp_std_csv, out_std, transform="gray", vae_path=VAE_PATH, batch=args.batch, workers=args.workers)
        if os.path.exists(tmp_std_csv):
            os.remove(tmp_std_csv)

    print(f"\n✓ 全部 15,548 样本编码完毕！总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
