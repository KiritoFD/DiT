import csv
import glob
import os

import numpy as np

for f in ("assets/eval_v13_strict_fixed.csv", "assets/eval_v13_seen_fixed.csv"):
    if not os.path.exists(f):
        print(f"  {f}: 不存在")
        continue
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    print(f"\n  === {f} ({len(rows)} 行) ===")
    print(f"    列: {list(rows[0].keys())}")
    for col in ("img_id", "old_50k_id"):
        if col in rows[0]:
            vals = [r[col] for r in rows[:4]]
            nums = [int(r[col]) for r in rows if r[col].strip()]
            print(f"    {col}: 前4={vals} 范围=[{min(nums)},{max(nums)}]")
    # image_path 的编号
    import re
    ipids = []
    for r in rows:
        m = re.search(r"(\d+)", os.path.basename(r["image_path"]))
        if m:
            ipids.append(int(m.group(1)))
    if ipids:
        print(f"    image_path 编号: 范围=[{min(ipids)},{max(ipids)}] "
              f"前4={ipids[:4]}")

# 训练 shards 的 id
ids = set()
for f in glob.glob("data/50k/shards_std_fixed/*.npz"):
    ids |= set(np.load(f)["img_ids"].tolist())
print(f"\n  训练 shards_std_fixed: {len(ids)} 个 id, 范围 "
      f"[{min(ids)}, {max(ids)}]")

# 检查 eval 的 id 是否在里面
for f in ("assets/eval_v13_strict_fixed.csv", "assets/eval_v13_seen_fixed.csv"):
    if not os.path.exists(f):
        continue
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    for col in ("img_id", "old_50k_id"):
        if col in rows[0]:
            v = set(int(r[col]) for r in rows if r[col].strip())
            print(f"  {os.path.basename(f)} 的 {col}: 不在训练 shards 的有 "
                  f"{len(v - ids)}/{len(v)}")
