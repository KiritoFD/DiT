#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
from collections import Counter

train = list(csv.DictReader(open("/root/Workspace/xy/DiT/assets/train.csv", encoding="utf-8")))
for sid, name in [(0, "楷"), (4, "隶")]:
    sub = [r for r in train if int(r["script_id"]) == sid]
    c = Counter(r["calligrapher"] for r in sub)
    print(f"=== {name} 全部书家({len(c)}个) 按行数 ===")
    for k, v in c.most_common():
        print(f"  {v:6d}  {k}")
    print()