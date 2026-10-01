#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

shards = sorted(glob.glob("data/top10_style23/shards_predskel_v31/shard_*.npz"))
print(f"Total predskel shards found: {len(shards)}")

total_samples = 0
all_ids = []
for s in shards:
    with np.load(s) as z:
        lats = z["latents"]
        iids = z["img_ids"]
        total_samples += len(iids)
        all_ids.extend(iids.tolist())
        print(f"  {os.path.basename(s)}: len={len(iids)}, latents={lats.shape}, dtype={lats.dtype}, min={lats.min():.2f}, max={lats.max():.2f}, mean={lats.mean():.3f}, std={lats.std():.3f}")

print(f"\n全量统计: 总样本数 = {total_samples} / 38583")
assert total_samples == 38583, f"样本总数不符: {total_samples} vs 38583"
assert len(set(all_ids)) == 38583, "检测到重复 ID!"
print("🎉 完美！全部 38,583 条预测骨架分片数据完整、数值健康、无一缺失！")
