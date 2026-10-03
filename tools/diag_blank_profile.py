#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_blank_profile.py — 空白样本的来源画像: 是否集中在某 source/书家/书体。"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
df = pd.read_csv("assets/train_top10_style23.csv")

fracs = np.zeros(len(df), dtype=np.float32)
for i, p in enumerate(df["image_path"]):
    if not os.path.exists(p):
        fracs[i] = -1
        continue
    a = np.asarray(Image.open(p).convert("L").resize((64, 64), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0
    fracs[i] = float((a < 0.5).mean())

df["ink_frac"] = fracs
bad = df[df["ink_frac"] < 0.005]
print("=" * 78)
print("[L] 空白样本 (%d 条) 来源画像" % len(bad))
print("=" * 78)
print("  idx 范围: %d .. %d  (占比区间: 是否集中在尾部? 全集 0..%d)" % (
    bad.index.min(), bad.index.max(), len(df) - 1))
print("  是否全部 idx >= 38000:", bool((bad.index >= 38000).all()))
print()
for col in ["source", "calligrapher", "script", "slot_name"]:
    if col in bad.columns:
        print("  by %-14s: %s" % (col, dict(bad[col].value_counts().head(8))))

print()
print("[M] 逐 1000-idx 桶的空白率 (定位是否某一段连续损坏):")
bins = np.arange(0, len(df) + 1000, 1000)
for b0 in bins[:-1]:
    seg = df[(df.index >= b0) & (df.index < b0 + 1000)]
    nb = int((seg["ink_frac"] < 0.005).sum())
    nlow = int((seg["ink_frac"] < 0.02).sum())
    flag = "  <== 异常段" if nb > 0 else ""
    print("   idx %6d-%6d: blank=%-3d  low(<2%%)=%-3d  mean_ink=%.4f%s" % (
        b0, b0 + 999, nb, nlow, seg["ink_frac"].clip(lower=0).mean(), flag))

print()
print("[N] 对照: 是否这些坏源的 src_image_path 也不存在/损坏")
n = 0
for _, r in bad.iterrows():
    if n >= 8:
        break
    src = r.get("src_image_path", "")
    ex = os.path.exists(str(src)) if src else None
    print("   idx=%d char='%s' source=%s src_exists=%s src=%s" % (
        r.name, r["character"], r.get("source"), ex, str(src)[:70]))
    n += 1
