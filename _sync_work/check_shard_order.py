import glob
import os

import numpy as np

files = sorted(glob.glob("data/50k/shards_std/shard_*.npz"))
print(f"  shard 文件数: {len(files)}")
tot = 0
for f in files:
    z = np.load(f)
    ids = z["img_ids"]
    print(f"   {os.path.basename(f)}: n={len(ids)}  id范围=[{ids.min()}, {ids.max()}]")
    tot += len(ids)
print(f"  总 latent: {tot}")

# 关键: 拼接后是否单调递增？若不是，sorted(文件名) 的顺序不等于 id 顺序
allids = np.concatenate([np.load(f)["img_ids"] for f in files])
print(f"\n  拼接后前10: {allids[:10]}")
print(f"  是否单调递增: {bool(np.all(np.diff(allids) > 0))}")
print(f"  重复 id 数: {len(allids) - len(set(allids.tolist()))}")
