#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/prepare_v22_glyph15k_dataset.py — 准备并校验 v22-glyph15k (已有书家掺字库) 的统一数据分片与软链接。

功能:
1. 在 data/50k_v2_glyph15k/ 下聚合建立统一的 shards_img, shards_std, shards_aux_skel3。
2. 软链接 50k 清洗底库 (50,786 纯碑帖) 与 15,548 张无跨体串扰的纯正 Glyph 补充 VAE 分片。
3. 校验端到端 66,334 条样本的连续性与完整性。
"""
import glob
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


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


def verify_shards(dst_dir, label):
    shards = sorted(glob.glob(os.path.join(dst_dir, "shard_*.npz")))
    seen_ids = set()
    total_samples = 0
    for sp in shards:
        d = np.load(sp)
        ids = d["img_ids"]
        for iid in ids:
            _iid = int(iid)
            if _iid in seen_ids:
                raise ValueError(f"Duplicate img_id={_iid} found in {sp} for {label}!")
            seen_ids.add(_iid)
        total_samples += len(ids)
        d.close()
    print(f"  ✓ {label}: {len(shards)} 分片, 累计 {total_samples} 条样本 (全部 ID 唯一且无碰撞)")
    return total_samples


def main():
    print("=== [1/3] 建立 v22-glyph15k (已有书家掺标准字库) 统一数据分片 ===")
    base_img_dirs = ["data/50k/shards_img", "data/supplements_glyph/shards_img"]
    base_std_dirs = ["data/50k/shards_std_fixed", "data/supplements_glyph/shards_std"]
    base_aux_dirs = ["data/50k/shards_aux_skel3", "data/supplements_glyph/shards_std"]

    dst_img = "data/50k_v2_glyph15k/shards_img"
    dst_std = "data/50k_v2_glyph15k/shards_std"
    dst_aux = "data/50k_v2_glyph15k/shards_aux_skel3"

    link_shards(base_img_dirs, dst_img)
    link_shards(base_std_dirs, dst_std)
    link_shards(base_aux_dirs, dst_aux)

    print("\n=== [2/3] 校验分片池样本总数 ===")
    c_img = verify_shards(dst_img, "目标图像 Latents (含评测池)")
    c_std = verify_shards(dst_std, "标准骨架 Latents (含评测池)")
    c_aux = verify_shards(dst_aux, "辅助形变骨架 Latents")

    print("\n=== [3/3] 运行 MCCDLatentDataset 端到端装载验证 (45 类已有书家) ===")
    from src.utils.latent_dataset import MCCDLatentDataset
    import json

    csv_path = "assets/train_50k_v2_augmented_glyph15k.csv"
    map_path = "assets/callig_id_map_50k.json"

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

    # 原始 66,334 扣除 8 个碑帖残损/缺字方框 (■, □) 之后为 66,326 条纯正 CJK 样本
    assert len(ds) == 66326, f"期望 66326 纯正样本，实际装载 {len(ds)}"

    test_indices = [0, len(ds)//2, 50785, 50786, len(ds)-1]
    for idx in test_indices:
        item = ds[idx]
        sample = ds.samples[idx]
        print(f"    Sample [{idx}]: char='{sample['character']}', callig='{sample['calligrapher']}' "
              f"(y_callig={item['y_callig'].item()}), img_shape={tuple(item['latent'].shape)}, "
              f"skel_shape={tuple(item['skel_latent'].shape)}")

    print("\n🎉 v22-glyph15k (已有书家掺标准字库) 数据分片链与 Dataset 装载验证 100% 通过！")


if __name__ == "__main__":
    main()

