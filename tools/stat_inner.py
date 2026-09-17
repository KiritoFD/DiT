# -*- coding: utf-8 -*-
"""stat_inner.py — 统计反色/黑框图 (inner 高) 的数量。"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

d = np.load("/tmp/scan_metrics.npz")
inn, w, ink, blob = d["inner"], d["w_max"], d["ink"], d["blob"]
rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
print(f"总 {len(inn)} 张")
print(f"inner: p50={np.percentile(inn,50):.3f} p90={np.percentile(inn,90):.3f} "
      f"p99={np.percentile(inn,99):.3f} max={inn.max():.3f}")
for t in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
    print(f"  inner>{t:<5} : {int((inn>t).sum()):6d}")
print(f"\n  w_max<=60 且 inner>0.9 : {int(((w<=60)&(inn>0.9)).sum())}")
print(f"  w_max<=60 且 inner>0.95: {int(((w<=60)&(inn>0.95)).sum())}")

sel = np.nonzero((w <= 60) & (inn > 0.9))[0]
print(f"\n  样例 (clean 判据下漏掉的反色/黑框图):")
for k in sel[:15]:
    r = rows[int(k)]
    print(f"    {r['image_path']} w={w[k]:5.1f} inner={inn[k]:.3f} "
          f"ink={ink[k]:.3f} blob={blob[k]:.3f} char={r.get('character','')} "
          f"{r.get('calligrapher','')}")

# 按数据源
from collections import Counter
c = Counter(rows[int(k)]["image_path"].split("/")[2] for k in sel)
print(f"\n  按数据源: {dict(c)}")
np.save("/tmp/inner_sel.npy", sel)
print(f"  -> /tmp/inner_sel.npy ({len(sel)} 个 index)")
