#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.utils.latent_dataset import MCCDLatentDataset

def main():
    print("=== [Preflight] 验证数据加载 ===")
    ds = MCCDLatentDataset(
        csv_file="assets/train_top10_style23.csv",
        latent_shards_dir="data/top10_style23/shards_img",
        img_root=None,
        skel_latent_shards_dir="data/top10_style23/shards_std",
        inst_skel_shards_dir="data/top10_style23/shards_aux_skel3",
        preload=False,
        load_image=False
    )
    print(f"  ✓ 数据集成功实例化，样本数: {len(ds):,}")
    item = ds[0]
    print(f"  ✓ 样本0: latent={tuple(item['latent'].shape)}, skel={tuple(item['skel_latent'].shape)}")

    print("\n=== [Preflight] 验证风格表 ===")
    d_emb = torch.load("assets/callig_script_emb_top10.pt", map_location="cpu", weights_only=False)
    emb = d_emb["embedding"] if isinstance(d_emb, dict) else d_emb
    print(f"  ✓ 风格表形状: {tuple(emb.shape)}")
    assert emb.shape == (23, 128), f"预期 (23, 128)，实际得到 {emb.shape}"

    print("\n=== [Preflight] 验证 SkelNet 权重 ===")
    d_skel = torch.load("assets/deform_skel_top10_v1.pt", map_location="cpu", weights_only=False)
    deform_sd = d_skel["deform"] if "deform" in d_skel else d_skel
    print(f"  ✓ SkelNet 参数量: {len(deform_sd)} 个键")

    print("\n=== [Preflight] 验证词表与映射 ===")
    from src.utils.callig_script_map import load_callig_script_map
    cmap = load_callig_script_map("assets/callig_script_id_map_top10.json")
    print(f"  ✓ 词表加载成功: num_pairs={cmap['num_pairs']}, num_calligraphers={cmap['num_calligraphers']}")

    print("\n🎉 预检 100% 成功，所有训练条件与组件全部就绪！")

if __name__ == "__main__":
    main()
