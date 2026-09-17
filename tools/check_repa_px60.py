# -*- coding: utf-8 -*-
"""check_repa_px60.py — 校验 REPA 的 DINO 特征缓存覆盖本数据集全部 img_id."""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = "data/dino_cache/base_sym_v1"
rows = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
want = {int(os.path.basename(r["image_path"])[:-4]) for r in rows}
print(f"数据集 id: {len(want)}")

ids = np.load(os.path.join(D, "ids.npy"))
print(f"缓存 ids: {ids.shape} dtype={ids.dtype}  (meta: "
      f"{json.load(open(os.path.join(D, 'meta.json'), encoding='utf-8'))})")
have = set(int(x) for x in ids)
miss = want - have
print(f"★ 缓存未覆盖的 id: {len(miss)}" + ("" if not miss else f"  e.g. {sorted(miss)[:10]}"))

feats = os.path.join(D, "feats.f16")
n, dim, patches = ids.shape[0], 384, 256
exp = n * patches * dim * 2
got = os.path.getsize(feats)
print(f"feats.f16 大小 {got/1e9:.2f}GB, 期望 {exp/1e9:.2f}GB -> "
      f"{'OK' if got == exp else '不匹配!'}")
