# -*- coding: utf-8 -*-
"""_cnt_csv.py — 统计各训练 CSV 的规模与数据源构成."""
import csv
import os
from collections import Counter

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
for f in ("assets/train_tongji_only.csv", "assets/train_base_noaug.csv",
          "assets/train_base_sym.csv", "assets/train_fame_tj_kxl.csv"):
    if not os.path.exists(f):
        print(f"{f}: 不存在")
        continue
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    c = Counter()
    for r in rows:
        p = r.get("image_path", "")
        parts = p.split("/")
        c[parts[2] if len(parts) > 2 else p] += 1
    print(f"\n{f}: rows={len(rows)}")
    print(f"  列: {list(rows[0].keys())}")
    for k, v in c.most_common():
        print(f"    {k:28s} {v}")
    if rows:
        r = rows[0]
        print(f"  样例: char={r.get('character')} script={r.get('script')} "
              f"callig={r.get('calligrapher')} -> {r.get('image_path','')[:70]}")
