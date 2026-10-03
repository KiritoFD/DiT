#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_scan_blank.py — 全量扫描: 找出空白/近空白图像样本 (与骨架无关的真实数据缺陷)。"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
df = pd.read_csv("assets/train_top10_style23.csv")

bad = []
inkstats = []
for i, r in df.iterrows():
    p = r["image_path"]
    if not os.path.exists(p):
        bad.append((i, r["img_id"], r["character"], "MISSING", 0.0))
        continue
    a = np.asarray(Image.open(p).convert("L").resize((64, 64), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0
    frac = float((a < 0.5).mean())
    inkstats.append(frac)
    if frac < 0.005:
        bad.append((i, r["img_id"], r["character"], "BLANK", frac))

inkstats = np.array(inkstats)
print("=" * 78)
print("[I] 全量 %d 样本图像墨迹占比统计" % len(inkstats))
print("=" * 78)
print("  mean=%.4f median=%.4f p1=%.4f p5=%.4f min=%.5f" % (
    inkstats.mean(), np.median(inkstats), np.percentile(inkstats, 1),
    np.percentile(inkstats, 5), inkstats.min()))
print("  墨迹 <0.5%% 的样本数: %d (%.3f%%)" % ((inkstats < 0.005).sum(), 100 * (inkstats < 0.005).mean()))
print("  墨迹 <2%%  的样本数: %d (%.3f%%)" % ((inkstats < 0.02).sum(), 100 * (inkstats < 0.02).mean()))
print()
print("[J] 空白/缺失样本清单 (共 %d 条, 仅列前 40):" % len(bad))
for b in bad[:40]:
    print("   idx=%d img_id=%d char='%s' %s frac=%.5f" % b)
print()
# 这些坏样本是否与骨架的字符匹配?
print("[K] 坏样本的骨架是否本身正确 (抽 5 个检查 std_path 有无墨迹):")
n = 0
for i, iid, ch, tag, frac in bad:
    if n >= 5:
        break
    sp = df.iloc[i]["std_path"]
    if os.path.exists(sp):
        b = np.asarray(Image.open(sp).convert("L").resize((64, 64), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0
        print("   idx=%d char='%s': image墨=%.4f  skel墨=%.4f -> %s" % (
            i, ch, frac, (b < 0.5).mean(),
            "骨架正常,是原图坏了" if (b < 0.5).mean() > 0.01 else "骨架也空"))
    n += 1
