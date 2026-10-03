import os, sys, csv, random
from collections import defaultdict, Counter
import numpy as np

random.seed(42)
np.random.seed(42)

input_csv = "assets/train_top10_style23.csv"
eval_out_csv = "assets/eval_top10_real_200.csv"
train_out_csv = "assets/train_top10_style23_real.csv"

with open(input_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} rows from {input_csv}")

# 1. Filter out synthetic data
real_rows = [r for r in rows if "font" not in r.get("source", "")]
syn_rows = [r for r in rows if "font" in r.get("source", "")]

print(f"Purged {len(syn_rows)} synthetic font samples.")
print(f"Retained {len(real_rows)} pure historical calligraphy samples.")

# 2. Check shard coverage
shards_gt_dir = "data/top10_style23/shards_gtskel_w7"
shards_std_dir = "data/top10_style23/shards_std_w7"

import glob
gt_shards = sorted(glob.glob(os.path.join(shards_gt_dir, "shard_*.npz")))
std_shards = sorted(glob.glob(os.path.join(shards_std_dir, "shard_*.npz")))

gt_ids = set()
for s in gt_shards:
    d = np.load(s)
    key = "img_ids" if "img_ids" in d else "ids"
    gt_ids.update(int(x) for x in d[key].tolist())

std_ids = set()
for s in std_shards:
    d = np.load(s)
    key = "img_ids" if "img_ids" in d else "ids"
    std_ids.update(int(x) for x in d[key].tolist())

print(f"GT w7 shard ids: {len(gt_ids)}, STD w7 shard ids: {len(std_ids)}")

valid_real_rows = []
missing = 0
for r in real_rows:
    iid = int(r["img_id"])
    if iid in gt_ids and iid in std_ids:
        valid_real_rows.append(r)
    else:
        missing += 1

print(f"Valid real rows with full w7 shard coverage: {len(valid_real_rows)} (missing: {missing})")

# 3. Stratified sampling of 200 evaluation samples across 23 slots
by_slot = defaultdict(list)
for r in valid_real_rows:
    by_slot[r["slot_name"]].append(r)

# Compute target allocations: total 200
# Min 2 samples per slot, rest proportional to sqrt(count) for balanced representation
slot_names = sorted(by_slot.keys())
weights = {s: np.sqrt(len(by_slot[s])) for s in slot_names}
total_w = sum(weights.values())

allocations = {}
remaining = 200
for s in slot_names:
    # guarantee at least 2 for tiny slots (or 1 if total is tiny, but min slot has 41)
    base = min(2, len(by_slot[s]))
    allocations[s] = base
    remaining -= base

# Distribute remaining based on weights
proportions = {s: weights[s] / total_w for s in slot_names}
sorted_by_rem = sorted(slot_names, key=lambda s: proportions[s], reverse=True)
idx = 0
while remaining > 0:
    s = sorted_by_rem[idx % len(sorted_by_rem)]
    if allocations[s] < len(by_slot[s]):
        allocations[s] += 1
        remaining -= 1
    idx += 1

print(f"\nAllocations across {len(slot_names)} slots (Total: {sum(allocations.values())}):")
eval_rows = []
train_rows = []

for s in slot_names:
    pool = by_slot[s]
    # shuffle with fixed seed
    random.shuffle(pool)
    n_eval = allocations[s]
    ev = pool[:n_eval]
    tr = pool[n_eval:]
    eval_rows.extend(ev)
    train_rows.extend(tr)
    print(f"  {s:16s}: Real Total={len(pool):5d} -> Eval={len(ev):3d}, Train={len(tr):5d}")

print(f"\nFinal Split:")
print(f"  Eval  set: {len(eval_rows):6d} pure real samples -> {eval_out_csv}")
print(f"  Train set: {len(train_rows):6d} pure real samples -> {train_out_csv}")

# 4. Save CSVs
fieldnames = list(rows[0].keys())

with open(eval_out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(eval_rows)

with open(train_out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train_rows)

print("\nVerification: Checking disjointness...")
eval_ids = set(r["img_id"] for r in eval_rows)
train_ids = set(r["img_id"] for r in train_rows)
overlap = eval_ids.intersection(train_ids)
print(f"Overlap between train and eval: {len(overlap)} (Must be 0!)")
assert len(overlap) == 0, "Data leakage detected!"
print("Data cleaning & stratified 200-eval split complete!")
