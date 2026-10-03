# -*- coding: utf-8 -*-
"""_diag_gmiss.py — 定位 std skel (g) 缺失 22.85% 的来源."""
import csv
import os
import re
import sys
from collections import Counter

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

key2uid = {}
for r in csv.DictReader(open("data/skel/std_skel3_base_key2uid.csv", encoding="utf-8")):
    key2uid[(r["script"], r["character"])] = int(r["uid"])
print(f"[base key2uid] {len(key2uid)}")

# tongji 独立 key2uid?
tj = {}
p = "data/skel/std_skel3_tj_key2uid.csv"
if os.path.exists(p):
    for r in csv.DictReader(open(p, encoding="utf-8")):
        tj[(r["script"], r["character"])] = int(r["uid"])
    print(f"[tj key2uid] {len(tj)}  列样例={list(csv.DictReader(open(p, encoding='utf-8')).fieldnames)}")

rows = list(csv.DictReader(open("assets/train_base_sym.csv", encoding="utf-8")))
miss_src, miss_aug, miss_ch, miss_scr = Counter(), Counter(), Counter(), Counter()
n_miss = 0
for r in rows:
    k = (r.get("script", ""), r.get("character", ""))
    if k in key2uid:
        continue
    n_miss += 1
    p2 = r.get("image_path", "").split("/")
    miss_src[p2[2] if len(p2) > 2 else "?"] += 1
    miss_aug[r.get("aug", "")] += 1
    miss_ch[r.get("character", "")] += 1
    miss_scr[r.get("script", "")] += 1

print(f"\n[缺失] {n_miss}/{len(rows)} = {100*n_miss/len(rows):.2f}%")
print(f"  按数据源: {miss_src.most_common()}")
print(f"  按 aug  : {miss_aug.most_common()}")
print(f"  按书体  : {miss_scr.most_common()}")
print(f"  缺失最多的字: {miss_ch.most_common(15)}")
print(f"  缺失唯一字数: {len(miss_ch)}")
if tj:
    cov = sum(1 for k in miss_ch if k in tj)
    print(f"  其中能用 tj key2uid 补上的字: {cov} 种")
