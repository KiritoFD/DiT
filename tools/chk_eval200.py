"""查 eval_real200_cache 的来历: 是否随机切/是否与训练集重叠。"""
import csv
import os
import re
import sys

import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
CACHE = "data/top10_style23/eval_real200_cache.pt"
TRAIN_CSV = "assets/train_top10_style23_real.csv"

c = th.load(CACHE, map_location="cpu", weights_only=False)
print(f"[cache] keys={list(c.keys())}")
for k in list(c.keys()):
    v = c[k]
    if hasattr(v, "shape"):
        print(f"   {k:<16} {tuple(v.shape)}  {v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"   {k:<16} list[{len(v)}]  样例={v[:3]}")
    else:
        print(f"   {k:<16} {type(v).__name__} = {v}")

# 找 cache 里可能存在的 id / 来源信息
ids = None
for k in ("img_ids", "ids", "id", "image_ids"):
    if k in c:
        ids = [int(x) for x in c[k]]
        print(f"[ids] 取自 cache['{k}'], n={len(ids)} 前 8 = {ids[:8]}")
        break
if ids is None:
    for k in ("rows", "meta", "paths", "src"):
        if k in c:
            print(f"[meta] cache['{k}'] 样例: {c[k][:3]}")
            break

# 训练集 id
tr = []
with open(TRAIN_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        m = re.search(r"(\d+)\.png$", r.get("image_path", "") or "")
        if m:
            tr.append(int(m.group(1)))
print(f"[train] {TRAIN_CSV} n={len(tr)} 前 8 = {tr[:8]}")

if ids is not None:
    s_tr, s_ev = set(tr), set(ids)
    inter = s_ev & s_tr
    print(f"\n[重叠] eval200 ∩ train = {len(inter)}/{len(ids)} "
          f"({len(inter) / max(len(ids), 1):.1%})  -> 训练集内占比")
    print(f"[范围] eval id min={min(ids)} max={max(ids)} | "
          f"train id min={min(tr)} max={max(tr)}")
