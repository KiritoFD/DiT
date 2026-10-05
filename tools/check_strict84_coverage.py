import os, sys, glob, csv, re
import numpy as np

csv_path = "assets/eval_top10_strict_subset84.csv"
shards_dir = "data/50k_v2_glyph15k/shards_std"

# Load shards index
shard_ids = {}
for p in sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz"))):
    with np.load(p) as z:
        for i, iid in enumerate(z["img_ids"]):
            shard_ids[int(iid)] = (p, i)

rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
print(f"Total rows in strict84 CSV: {len(rows)}")

matched_old = 0
matched_img = 0
for r in rows:
    # Try old_50k_id
    old_id = int(r.get("old_50k_id", -1))
    if old_id in shard_ids:
        matched_old += 1
    # Try image_path id
    m = re.search(r"(\d+)\.png", r["image_path"])
    if m and int(m.group(1)) in shard_ids:
        matched_img += 1

print(f"Matched by old_50k_id: {matched_old}/{len(rows)}")
print(f"Matched by image_path: {matched_img}/{len(rows)}")
