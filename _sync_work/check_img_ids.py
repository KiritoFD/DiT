import csv

import numpy as np

z = np.load("data/50k/shards_std/shard_00000.npz")
ids = z["img_ids"]
print(f"  ids dtype: {ids.dtype}")
print(f"  前5: {ids[:5]}")
print(f"  后5: {ids[-5:]}")
print(f"  总数: {len(ids)}")
print()

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
print(f"  csv 行数: {len(rows)}")
print(f"  csv 前5 img_id: {[r['img_id'] for r in rows[:5]]}")
print(f"  csv 前5 character: {[r['character'] for r in rows[:5]]}")
print(f"  csv 前5 image_path: {[r['image_path'][-30:] for r in rows[:5]]}")

# 关键测试: csv[10] 的字是什么？shard[10] 的 img_id 是什么？
print()
print("--- 几个样本的对照 ---")
for i in [0, 1, 10, 100, 1000, 5000, 50000]:
    if i < len(rows):
        cr = rows[i]
        sid = ids[i] if i < len(ids) else "OUT OF RANGE"
        print(f"  csv[{i}]: char={cr['character']!r}  img_path={cr['image_path'][-25:]}  shard[{i}].img_id={sid}")