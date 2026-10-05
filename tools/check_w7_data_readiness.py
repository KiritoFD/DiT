import os, sys, glob, csv
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

print("=== 检查 w7 数据完备性 ===")

# 1. 训练目标: shards_gtskel_w7
train_gt_shards = sorted(glob.glob("data/top10_style23/shards_gtskel_w7/shard_*.npz"))
print(f"1. 训练目标 (shards_gtskel_w7): 找到 {len(train_gt_shards)} 个分片")
total_train_gt = 0
for p in train_gt_shards:
    with np.load(p) as z:
        total_train_gt += len(z["img_ids"])
print(f"   总样本数: {total_train_gt}")

# 2. 训练条件: shards_std_w7
train_std_shards = sorted(glob.glob("data/top10_style23/shards_std_w7/shard_*.npz"))
print(f"2. 训练条件 (shards_std_w7): 找到 {len(train_std_shards)} 个分片")
total_train_std = 0
for p in train_std_shards:
    with np.load(p) as z:
        total_train_std += len(z["img_ids"])
print(f"   总样本数: {total_train_std}")

# 3. 评测集 GT 骨架 (gt_skel_eval_strict84)
eval_gt_pngs = sorted(glob.glob("data/top10_style23/gt_skel_eval_strict84_png/*.png"))
print(f"3. 评测集 Strict84 GT PNG: 找到 {len(eval_gt_pngs)} 张图")
if eval_gt_pngs:
    im = np.asarray(Image.open(eval_gt_pngs[0]).convert("L")) < 128
    print(f"   样本 0 墨像素比例: {im.mean():.4f}")

# 4. 检查是否有 gt_skel_png_w7
gt_w7_pngs = sorted(glob.glob("data/top10_style23/gt_skel_png_w7/*.png"))
print(f"4. 真迹 w7 全量 PNG: 找到 {len(gt_w7_pngs)} 张图")
if gt_w7_pngs:
    im7 = np.asarray(Image.open(gt_w7_pngs[0]).convert("L")) < 128
    print(f"   样本 0 w7 墨像素比例: {im7.mean():.4f}")
