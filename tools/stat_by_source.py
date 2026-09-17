# -*- coding: utf-8 -*-
"""stat_by_source.py — 按数据源分层统计当前数据集, 并定位污渍图."""
import csv
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 原始各源规模 (来自 train_base_noaug.csv)
orig = Counter()
orig_char = {}
for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
    s = r["image_path"].split("/")[2]
    orig[s] += 1

rows = list(csv.DictReader(open("assets/train_clean_v1_final.csv", encoding="utf-8")))
cur = Counter()
for r in rows:
    s = r["image_path"].split("/")[2] if r["image_path"].count("/") > 2 else "?"
    cur[s] += 1

print("=== 分层存活 ===")
for s in sorted(orig, key=lambda x: -orig[x]):
    o, c = orig[s], cur.get(s, 0)
    print(f"  {s:22s} 原始 {o:6d} -> 现存 {c:6d}  ({100*c/o:5.1f}%)")
print(f"  {'合计':22s} 原始 {sum(orig.values()):6d} -> 现存 {sum(cur.values()):6d}")

print(f"\n=== 当前 {len(rows)} 行的 script 分布 ===")
print(f"  {dict(Counter(r.get('script','') for r in rows))}")

# 定位 glitch: 用 p90 背景亮度找脏背景
import multiprocessing as mp

import numpy as np
from PIL import Image


def probe(r):
    try:
        a = np.asarray(Image.open(r["image_path"]).convert("L"))
    except Exception:
        return (r["image_path"], r.get("character", ""), -1.0)
    b = a[a > 128]
    return (r["image_path"], r.get("character", ""),
            float(np.percentile(b, 50)) if b.size > 100 else 0.0)


with mp.Pool(40) as pool:
    res = pool.map(probe, rows, chunksize=256)
bg = np.array([x[2] for x in res])
print(f"\n=== 背景(亮部中位)分布 ===")
print(f"  p1={np.percentile(bg,1):.1f} p5={np.percentile(bg,5):.1f} "
      f"p50={np.percentile(bg,50):.1f} min={bg.min():.1f}")
for t in (200, 220, 235, 245, 250):
    n = int((bg < t).sum())
    print(f"  bg<{t}: {n:6d} ({100*n/len(bg):5.2f}%)")
worst = sorted(res, key=lambda x: x[2])[:12]
print("\n  背景最脏的 12 张:")
for p, ch, v in worst:
    src = p.split("/")[2] if p.count("/") > 2 else "?"
    print(f"    bg={v:6.1f} {src:16s} char={ch}  {p}")
