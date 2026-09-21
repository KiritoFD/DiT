import csv
import glob
import os
import re

import numpy as np

rows = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                                encoding="utf-8")))
print(f"  列: {list(rows[0].keys())}")
print(f"  行数: {len(rows)}")

for col in ("img_id", "old_50k_id"):
    if col in rows[0]:
        vals = [r[col] for r in rows[:5]]
        print(f"  {col} 前5: {vals}")
        try:
            nums = [int(r[col]) for r in rows if r[col].strip()]
            print(f"    {col}: n={len(nums)} 范围=[{min(nums)}, {max(nums)}] "
                  f"唯一={len(set(nums))}")
        except Exception as e:
            print(f"    {col} 非整数: {e}")

# 原 shards_std 的 img_id
ids_old = set()
for f in glob.glob("data/50k/shards_std/*.npz"):
    ids_old |= set(np.load(f)["img_ids"].tolist())
print(f"\n  原 shards_std img_id: {len(ids_old)} 个, 范围 "
      f"[{min(ids_old)}, {max(ids_old)}]")

# 我新建的
ids_new = set()
for f in glob.glob("data/50k/shards_std_fixed/*.npz"):
    ids_new |= set(np.load(f)["img_ids"].tolist())
print(f"  新 shards_std_fixed img_id: {len(ids_new)} 个, 范围 "
      f"[{min(ids_new)}, {max(ids_new)}]")

# 对比 old_50k_id 集合
if "old_50k_id" in rows[0]:
    oldids = set(int(r["old_50k_id"]) for r in rows if r["old_50k_id"].strip())
    print(f"\n  csv old_50k_id: {len(oldids)} 个")
    print(f"    不在原 shards_std: {len(oldids - ids_old)}")
    print(f"    不在新 shards_fixed: {len(oldids - ids_new)}")
    print(f"    -> 原 shards_std 与 old_50k_id {'匹配' if len(oldids-ids_old)==0 else '不匹配'}")

# std_path 编号集合
stdids = set()
for r in rows:
    m = re.search(r"(\d+)", os.path.basename(r["std_path"]))
    if m:
        stdids.add(int(m.group(1)))
print(f"\n  csv std_path 编号: {len(stdids)} 个")
print(f"    不在新 shards_fixed: {len(stdids - ids_new)}")
