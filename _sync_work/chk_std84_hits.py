"""strict84 csv 的 id 在各条件目录里的命中率 —— 定位"零条件"到底出在哪。"""
import csv
import glob
import os
import re

CSV = "assets/eval_v13_strict84_aligned.csv"
DIRS = ["data/top10_style23/predskel_std84_e2e",
        "data/top10_style23/shards_std",
        "data/50k_v2_glyph15k/shards_std",
        "data/top10_style23/gt_skel_eval_strict84",
        "data/top10_style23/shards_img_eval84"]

ids = []
with open(CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        m = re.search(r"(\d+)\.png$", r.get("image_path", "") or "")
        if m:
            ids.append(int(m.group(1)))
print(f"csv 行数={len(ids)} 前 5 个 id={ids[:5]}")
for d in DIRS:
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not fs:
        print(f"  {d}: 目录/分片不存在")
        continue
    have = set()
    for f in fs:
        import numpy as np
        with np.load(f) as z:
            have.update(int(i) for i in z["img_ids"])
    hit = sum(1 for i in ids if i in have)
    print(f"  {d}: 命中 {hit}/{len(ids)}  (分片 {len(fs)}, 共 {len(have)} 个 id)")
