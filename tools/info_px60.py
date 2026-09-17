# -*- coding: utf-8 -*-
"""info_px60.py — 一次性打印 px60 数据集与训练依赖的环境信息."""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=== callig_id_map ===")
p = "assets/callig_id_map_base.json"
m = json.load(open(p, encoding="utf-8"))
print(type(m).__name__, len(m))
items = list(m.items())[:6] if isinstance(m, dict) else m[:6]
print(items)

print("\n=== 训练 csv 对比 ===")
import csv
for f in ("assets/train_base_noaug.csv", "assets/train_base_sym_clean.csv",
          "assets/train_base_clean.csv", "assets/train_fame-kxl-tj-px60.csv"):
    if os.path.exists(f):
        n = sum(1 for _ in open(f, encoding="utf-8")) - 1
        print(f"  {f}: {n} 行")
    else:
        print(f"  {f}: 不存在")

print("\n=== sym_clean 的 aug 分布 / id 段 ===")
rows = list(csv.DictReader(open("assets/train_base_sym_clean.csv", encoding="utf-8")))
from collections import Counter
print("  aug:", dict(Counter(r.get("aug", "") for r in rows)))
print("  列:", list(rows[0].keys()))
print("  样例:", rows[0])
ids = [os.path.basename(r["image_path"])[:-4] for r in rows]
print(f"  id: n={len(ids)} unique={len(set(ids))}")

print("\n=== 已有 aug 图 (data/imgs/base_sym) ===")
n_sym = len(glob.glob("data/imgs/base_sym/*.png"))
print(f"  {n_sym} png")

print("\n=== noaug 全量行号 == index? ===")
naug = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
print(f"  noaug {len(naug)} 行")
print("  csv 是否带 idx 列:", "idx" in naug[0])

print("\n=== dino cache (REPA) ===")
for d in glob.glob("data/dino_cache/*"):
    fs = glob.glob(os.path.join(d, "*"))
    print(f"  {d}: {len(fs)} 项")
    if fs:
        print("     e.g.", os.path.basename(fs[0]))

print("\n=== 现有 latent shards ===")
for d in ("data/latents/final_latents_base_sym", "data/skel/std_skel3_latents_base_sym",
          "data/fame-kxl-tj-px60/shards_img", "data/fame-kxl-tj-px60/shards_std"):
    if os.path.isdir(d):
        fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
        t = 0
        for f in fs:
            with np.load(f) as z:
                t += z["img_ids"].shape[0]
        print(f"  {d}: {len(fs)} shards, {t} latents")
    else:
        print(f"  {d}: 不存在")
