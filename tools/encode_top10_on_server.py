#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/encode_top10_on_server.py — 在 4090 上编码 top10_style23 的全量 VAE Latents (Img, Std, Aux)"""
import os
import sys
import csv
import time
import argparse

sys.stdout.reconfigure(encoding="utf-8")

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.data.vae_io import encode_csv

CSV_PATH = "assets/train_top10_style23.csv"
OUT_BASE = "data/top10_style23"
VAE_PATH = "pretrained_models/sd-vae-ft-ema"

def main():
    parser = argparse.ArgumentParser(description="在 4090 上编码 Top 10 的 VAE 分片")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target", default="all", choices=["img", "std", "aux", "all"])
    args = parser.parse_args()

    print(f"=== 开始在 4090 上编码 Top 10 (38,583 样本) VAE 潜变量分片 ===")
    t_start = time.time()

    # 1. 目标图像分片 (shards_img)
    if args.target in ("img", "all"):
        out_img = os.path.join(OUT_BASE, "shards_img")
        print(f"\n[1/3] 编码目标图像 VAE Latents -> {out_img} ...")
        t0 = time.time()
        encode_csv(CSV_PATH, out_img, transform="gray", vae_path=VAE_PATH, batch=args.batch, workers=args.workers)
        print(f"  ✓ shards_img 完成，耗时: {time.time()-t0:.1f}s")

    # 2. 标准骨架分片 (shards_std)
    if args.target in ("std", "all"):
        out_std = os.path.join(OUT_BASE, "shards_std")
        print(f"\n[2/3] 编码标准骨架 VAE Latents -> {out_std} ...")
        tmp_std_csv = "assets/_tmp_top10_std.csv"
        with open(CSV_PATH, encoding="utf-8") as f, open(tmp_std_csv, "w", encoding="utf-8", newline="") as fo:
            reader = csv.DictReader(f)
            writer = csv.writer(fo)
            writer.writerow(["image_path"])
            for r in reader:
                writer.writerow([r["std_path"]])
        t0 = time.time()
        encode_csv(tmp_std_csv, out_std, transform="gray", vae_path=VAE_PATH, batch=args.batch, workers=args.workers)
        if os.path.exists(tmp_std_csv):
            os.remove(tmp_std_csv)
        print(f"  ✓ shards_std 完成，耗时: {time.time()-t0:.1f}s")

    # 3. 真实墨迹 3px 骨架分片 (shards_aux_skel3, 用于 SkelNet 形变监督)
    if args.target in ("aux", "all"):
        out_aux = os.path.join(OUT_BASE, "shards_aux_skel3")
        print(f"\n[3/3] 编码真实墨迹 3px 骨架 VAE Latents -> {out_aux} ...")
        t0 = time.time()
        encode_csv(CSV_PATH, out_aux, transform="skel", skel_dilate=1, vae_path=VAE_PATH, batch=args.batch, workers=args.workers)
        print(f"  ✓ shards_aux_skel3 完成，耗时: {time.time()-t0:.1f}s")

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"🎉 全部 VAE 编码任务圆满完成！总耗时: {total_time/60:.2f} 分钟 ({total_time:.1f}s)")
    print("=" * 65)

if __name__ == "__main__":
    main()
