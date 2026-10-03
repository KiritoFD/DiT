#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.utils.latent_dataset import MCCDLatentDataset

def main():
    print("=== [Preload Test] 验证全量内存预加载 ===")
    ds = MCCDLatentDataset(
        csv_file="assets/train_top10_style23.csv",
        latent_shards_dir="data/top10_style23/shards_img",
        img_root=None,
        skel_latent_shards_dir="data/top10_style23/shards_std",
        inst_skel_shards_dir="data/top10_style23/shards_aux_skel3",
        preload=True,
        load_image=False
    )
    print(f"🎉 预加载成功！数据集总数: {len(ds):,}")
    item = ds[0]
    print(f"  ✓ 样本0: latent={tuple(item['latent'].shape)}, skel={tuple(item['skel_latent'].shape)}")

if __name__ == "__main__":
    main()
