#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_verify_std_png.py — 直接核对 std_path PNG 与 image_path PNG 字形一致性,
绕开 VAE,排除 latent 编解码误差。
同时检查疑似错配的 idx 5000/20000/38582。
"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

df = pd.read_csv("assets/train_top10_style23.csv")


def load_gray(p, size=64):
    im = Image.open(p).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    return a


def ink(a, t=0.5):
    return a < t


print("=" * 92)
print("[G] 直接核对 CSV 中 image_path 与 std_path 的 PNG (不走 VAE)")
print("=" * 92)
print("  %s | %s | %s | %s | img墨%% | std墨%% | std覆盖率 | 像素IoU | 判定" % (
    "idx".rjust(6), "img_id".rjust(5), "calligrapher·script".ljust(14),
    "char".ljust(4)))

test_idx = [0, 1, 2, 3, 4, 50, 500, 1000, 5000, 20000, 38580, 38582]
for idx in test_idx:
    r = df.iloc[idx]
    ip, sp = r["image_path"], r["std_path"]
    if not os.path.exists(ip) or not os.path.exists(sp):
        print("%6d | MISSING ip=%s sp=%s" % (idx, os.path.exists(ip), os.path.exists(sp)))
        continue
    a = load_gray(ip); b = load_gray(sp)
    ia, ib = ink(a), ink(b)
    # 骨架是细线, 用"膨胀后覆盖率": 骨架像素落在图像墨迹附近的比例
    from scipy.ndimage import binary_dilation
    ia_dil = binary_dilation(ia, iterations=3)
    cover = (ib & ia_dil).sum() / max(1, ib.sum())
    iou = (ia & ib).sum() / max(1, (ia | ib).sum())
    verdict = "OK-同字" if cover > 0.55 else ("?? 骨架与图像偏离" if cover > 0.25 else "!! 疑似错配")
    print("%6d | %5d | %-14s | %-4s | %5.1f%% | %5.1f%% | %6.1f%% | %6.3f | %s" % (
        idx, r["img_id"], "%s·%s" % (r["calligrapher"], r["script"]), r["character"],
        100 * ia.mean(), 100 * ib.mean(), 100 * cover, iou, verdict))

print()
print("=" * 92)
print("[H] 全量统计: 随机 2000 样本的骨架覆盖率分布 (判定整体是否干净)")
print("=" * 92)
from scipy.ndimage import binary_dilation
rng = np.random.default_rng(0)
idxs = rng.choice(len(df), size=2000, replace=False)
covers = []
for idx in idxs:
    r = df.iloc[idx]
    try:
        a = load_gray(r["image_path"]); b = load_gray(r["std_path"])
    except Exception:
        continue
    ia, ib = ink(a), ink(b)
    if ib.sum() < 5:
        continue
    ia_dil = binary_dilation(ia, iterations=3)
    covers.append((ib & ia_dil).sum() / ib.sum())
covers = np.array(covers)
print("  样本数:", len(covers))
print("  覆盖率 mean=%.3f  median=%.3f  p10=%.3f  p25=%.3f  p75=%.3f" % (
    covers.mean(), np.median(covers), np.percentile(covers, 10),
    np.percentile(covers, 25), np.percentile(covers, 75)))
print("  覆盖率 >0.55 比例: %.1f%%" % (100 * (covers > 0.55).mean()))
print("  覆盖率 >0.70 比例: %.1f%%" % (100 * (covers > 0.70).mean()))
print("  覆盖率 <0.25 比例: %.1f%%  (若显著>0 则存在错配/空洞字)" % (100 * (covers < 0.25).mean()))
