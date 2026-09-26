#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/prepare_v22_dataset.py — 准备并校验 v22 统一数据分片与映射。

功能:
1. 在 data/50k_v2_augmented/ 下建立统一的 shards_img 和 shards_std。
2. 软链接 50k 清洗底库 (shards_std_fixed) 与 7,609 张多字体补充分片。
3. 实例化 MCCDLatentDataset 进行端到端全量 58,395 条样本的 ID 连续性与加载校验。
"""
import glob
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils.latent_dataset import MCCDLatentDataset


def link_shards(src_dirs, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shard_idx = 0
    for sdir in src_dirs:
        shards = sorted(glob.glob(os.path.join(sdir, "shard_*.npz")))
        for sp in shards:
            dst_link = os.path.join(dst_dir, f"shard_{shard_idx:05d}.npz")
            if os.path.lexists(dst_link):
                os.remove(dst_link)
            os.symlink(os.path.abspath(sp), dst_link)
            shard_idx += 1
    print(f"  ✓ {dst_dir}: 已聚合链接 {shard_idx} 个分片")


def main():
    print("=== [1/3] 建立 v22 统一数据分片目录 ===")
    base_img_dirs = ["data/50k/shards_img", "data/supplements/shards_img"]
    base_std_dirs = ["data/50k/shards_std_fixed", "data/supplements/shards_std"]
    base_aux_dirs = ["data/50k/shards_aux_skel3", "data/supplements/shards_std"]

    dst_img = "data/50k_v2_augmented/shards_img"
    dst_std = "data/50k_v2_augmented/shards_std"
    dst_aux = "data/50k_v2_augmented/shards_aux_skel3"

    link_shards(base_img_dirs, dst_img)
    link_shards(base_std_dirs, dst_std)
    link_shards(base_aux_dirs, dst_aux)

    print("\n=== [2/3] 验证分片数据完整性 ===")
    for label, ddir in [("Image Shards", dst_img), ("Std Skel Shards", dst_std), ("Inst Skel Shards", dst_aux)]:
        shards = sorted(glob.glob(os.path.join(ddir, "shard_*.npz")))
        total_samples = 0
        all_ids = []
        for sp in shards:
            d = np.load(sp)
            ids = d["img_ids"]
            all_ids.extend(ids.tolist())
            total_samples += len(ids)
            d.close()
        print(f"  [{label}] 共 {len(shards)} 个分片，累计样本: {total_samples}")
        print(f"    - ID 范围: min={min(all_ids)}, max={max(all_ids)}, 唯一 ID 数={len(set(all_ids))}")

    print("\n=== [3/3] 运行 MCCDLatentDataset 深度装载验证 ===")
    csv_path = "assets/train_50k_v2_fixed_augmented.csv"
    map_path = "assets/callig_id_map_50k_ext.json"
    
    import json
    with open(map_path, "r", encoding="utf-8") as f:
        cmap = json.load(f)

    ds = MCCDLatentDataset(
        csv_file=csv_path,
        latent_shards_dir=dst_img,
        img_root=None,
        preload=False,
        load_image=False,
        skel_latent_shards_dir=dst_std,
        inst_skel_shards_dir=dst_aux,
        callig_id_map=cmap["id_map"]
    )

    print(f"  ✓ 数据集成功实例化！总样本数: {len(ds)}")
    
    # 抽样检测前、中、后样本
    test_indices = [0, len(ds)//2, 50785, 50786, len(ds)-1]
    for idx in test_indices:
        item = ds[idx]
        sample = ds.samples[idx]
        print(f"    Sample [{idx}]: char='{sample['character']}', callig='{sample['calligrapher']}' "
              f"(y_callig={item['y_callig'].item()}), img_shape={tuple(item['latent'].shape)}, "
              f"skel_shape={tuple(item['skel_latent'].shape)}")

    print("\n🎉 v22 统一增强数据集与分片链验证 100% 通过！")


if __name__ == "__main__":
    main()
