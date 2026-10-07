#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/probe_intraclass_pixelpairs.py — 真迹内在方差的**像素域**版本 (CPU, 不需 GPU)。

为什么要有这一版:
  tools/probe_theoretical_bounds.py 的 0.6588 是先把两张真迹各自 VAE 编解码一次、
  再在**解码后**的图上比的 (dec1 vs dec2)。所以那 0.6588 **已经内含 VAE 损耗**。
  本脚本在**原始像素**上比同一批对, 用来:
    (a) 给出"两位书法家写同一个字, 像素相似度多少"这个人类可读的参照;
    (b) 与解码版相减, 直接量出 SD-VAE 在这套指标上的代价 ——
        顺便裁决 DOC-105-04(自重构 SSIM≈0.82) 与 DOC-105-05(实测 0.9729) 的矛盾。

抽对协议与 probe_theoretical_bounds.py **完全一致** (random.seed(42) 打乱键, 取前 300 个
键, 每键取 CSV 里出现的前两行), 这样两次结果可直接相减。

用法: PYTHONPATH=. python tools/probe_intraclass_pixelpairs.py [--pairs 300]
"""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.metrics import ssim as our_ssim          # noqa: E402
from src.eval.metrics_ink import ink_iou as our_ink_iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--pairs", type=int, default=300)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["calligrapher"], r["script"], r["character"])].append(r)
    multis = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"样本 {len(rows):,} | 唯一(c,s,ch) {len(groups):,} | >=2 张的键 {len(multis):,}")

    random.seed(42)                     # 与 probe_theoretical_bounds.py 同种子
    keys = list(multis.keys())
    random.shuffle(keys)

    ss, iou, used, miss = [], [], 0, 0
    for k in keys:
        if used >= a.pairs:
            break
        r1, r2 = multis[k][0], multis[k][1]
        p1, p2 = r1["image_path"], r2["image_path"]
        if not (os.path.exists(p1) and os.path.exists(p2)):
            miss += 1
            continue
        with Image.open(p1) as im:
            x1 = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(p2) as im:
            x2 = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        ss.append(float(our_ssim(x1, x2)))
        iou.append(float(our_ink_iou(x1, x2)))
        used += 1

    ss, iou = np.array(ss), np.array(iou)
    print("\n" + "=" * 74)
    print(f"  真迹内在方差 · **像素域** (未过 VAE)   N={len(ss)} 对   缺图 {miss}")
    print("=" * 74)
    print(f"  SSIM  均值 {ss.mean():.4f}   中位数 {np.median(ss):.4f}   "
          f"std {ss.std():.4f}   min {ss.min():.4f}   max {ss.max():.4f}")
    print(f"  墨迹IoU 均值 {iou.mean():.4f}   中位数 {np.median(iou):.4f}")
    print("=" * 74)
    print("  参照 (DOC-105-04, 同为前两行抽对, 但**经 VAE 解码后**再比):")
    print("    SSIM 均值 0.6588 / 中位数 0.6490")
    print(f"  → 本轮像素域均值 {ss.mean():.4f} 与解码版 0.6588 的差 = "
          f"{ss.mean()-0.6588:+.4f}  (负数=VAE 让两张真迹**更不像**; 正数=解码反而更接近)")


if __name__ == "__main__":
    main()
