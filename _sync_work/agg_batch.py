#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 in_mem_eval 的 batch CSV: 按 set 报 n / ssim / frag 均值"""
import collections
import csv
import sys

path = sys.argv[1]
m = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(path, encoding="utf-8")):
    for k in ("ssim", "ink_ssim", "frag_ratio"):
        try:
            m[r["set"]][k].append(float(r[k]))
        except (KeyError, ValueError):
            pass
for s, d in m.items():
    parts = []
    for k in ("ssim", "ink_ssim", "frag_ratio"):
        if d[k]:
            parts.append(f"{k}={sum(d[k])/len(d[k]):.4f}")
    print(f"  {s}: n={len(d['ssim'])} " + " ".join(parts))
