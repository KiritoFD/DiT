# -*- coding: utf-8 -*-
"""plan_px60.py — 起训前的依赖核查 (只读, 不改任何东西).

A. 书家表 callig_id_map_base.json 是否覆盖本数据集的全部 calligrapher_id
B. 增强图 data/imgs/base_sym/{7000000|7100000 + noaug_idx}.png 是否齐全
C. REPA 的 DINO 特征缓存 (repa_cache_dir) 结构 + 对本数据集 id 的覆盖率
D. 新 encode 的 shards_img 与旧 final_latents_base_sym 对同 id 是否一致
   (一致 => 旧库可直接复用, 不必重编码)
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME = "fame-kxl-tj-px60"
CSV = f"assets/train_{NAME}.csv"
OLD_IMG_LAT = "data/latents/final_latents_base_sym"
DINO_DIR = "data/dino_cache/base_sym_v1"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"[{NAME}] base rows = {len(rows)}")

# ---- A. 书家表 ----
m = json.load(open("assets/callig_id_map_base.json", encoding="utf-8"))
idmap = m["id_map"]
raw = {r["calligrapher_id"] for r in rows}
miss = sorted(raw - set(idmap))
print(f"\nA. 书家表: num_calligraphers={m['num_calligraphers']}, id_map={len(idmap)}")
print(f"   本数据集 calligrapher_id {len(raw)} 个, 未覆盖 {len(miss)} {miss[:10]}")
cov = {(r['calligrapher'], r['calligrapher_id']) for r in rows}
print(f"   书家(名,id) 对 {len(cov)} 个, 缺 id 的: "
      f"{sum(1 for _, i in cov if i not in idmap)}")

# ---- B. 增强图是否齐全 ----
naug = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
b2i = {os.path.basename(r["image_path"])[:-4]: k for k, r in enumerate(naug)}
print(f"\nB. noaug 索引: {len(naug)} 行, basename->idx {len(b2i)} 个")
idx_of = []
for r in rows:
    b = os.path.basename(r["image_path"])[:-4]
    idx_of.append(b2i.get(b))
print(f"   本数据集能定位 noaug idx 的: {sum(1 for x in idx_of if x is not None)}/{len(rows)}")
have_t = have_n = 0
for k in idx_of:
    if k is None:
        continue
    if os.path.exists(f"data/imgs/base_sym/{7000000 + k}.png"):
        have_t += 1
    if os.path.exists(f"data/imgs/base_sym/{7100000 + k}.png"):
        have_n += 1
print(f"   增强图存在: tp {have_t}/{len(rows)}  tn {have_n}/{len(rows)}")
print(f"   => 增强后行数约 {len(rows) + have_t + have_n}")

# ---- C. DINO 缓存 ----
print(f"\nC. DINO 缓存 {DINO_DIR}:")
if os.path.isdir(DINO_DIR):
    for f in sorted(os.listdir(DINO_DIR)):
        p = os.path.join(DINO_DIR, f)
        print(f"   {f}  {os.path.getsize(p)/1e6:.1f}MB")
    mp = os.path.join(DINO_DIR, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
        print(f"   meta: {json.dumps(meta, ensure_ascii=False)[:600]}")

# ---- D. latent 一致性 ----
print(f"\nD. latent 一致性 (新 {NAME}/shards_img  vs 旧 {OLD_IMG_LAT}):")
def load_map(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            ids = z["img_ids"]
            for j, i in enumerate(ids):
                mp[int(i)] = (sp, j)
    return mp

new_map = load_map(f"data/{NAME}/shards_img")
old_map = load_map(OLD_IMG_LAT)
print(f"   new {len(new_map)} ids, old {len(old_map)} ids, 交集 {len(set(new_map) & set(old_map))}")
samp = sorted(new_map)[:8]
mx = 0.0
for i in samp:
    with np.load(new_map[i][0]) as z:
        a = z["latents"][new_map[i][1]]
    with np.load(old_map[i][0]) as z:
        b = z["latents"][old_map[i][1]]
    d = float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max())
    mx = max(mx, d)
    print(f"   id={i}: max|Δ|={d:.6f}")
print(f"   => 最大偏差 {mx:.6f}  ({'一致, 旧库可复用' if mx < 1e-2 else '不一致!'})")
