"""查 REPA 的 DINO 缓存是否覆盖当前数据集的 img_id（不覆盖 -> 静默用错特征）。"""
import csv
import json
import os
import re

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

CACHE = "data/dino_cache/base_sym_v1"
CSV = "assets/train_fame-kxl-tj-px60.csv"

ids = np.load(os.path.join(CACHE, "ids.npy"))
print(f"  缓存 ids : n={len(ids):,}  min={ids.min()}  max={ids.max()}")
print(f"  meta     : {json.load(open(os.path.join(CACHE, 'meta.json'), encoding='utf-8'))}")

ds = []
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    m = re.search(r"(\d+)\.png$", r["image_path"])
    if m:
        ds.append(int(m.group(1)))
ds = np.array(ds)
have = np.isin(ds, ids)
print()
print(f"  数据集   : n={len(ds):,}  min={ds.min()}  max={ds.max()}")
print(f"  缓存命中 : {have.sum():,}/{len(ds):,} ({100 * have.mean():.1f}%)")
miss = ds[~have]
if len(miss):
    print(f"  未命中   : {miss[:10].tolist()} ...")
else:
    print("  未命中   : 无 —— 全覆盖")
