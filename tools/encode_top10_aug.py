#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""encode_top10_aug.py — 为 top10 增强数据集 (77,823 图) 批量生成 VAE latent shards (支持断点续编)

输入: /home/ds/Workspace/moyi/exp-std-csv/train_top10_aug_sym.csv
图像基准目录: /home/ds/Workspace/DiT
输出目录: /home/ds/Workspace/moyi/data/top10_style23/shards_img_aug/
产物: shard_%05d.npz, keys: ['latents' (N, 4, 32, 32) float16, 'img_ids' (N,)]
"""

import csv
import glob
import os
import sys
import time
from diffusers.models import AutoencoderKL
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = "/home/ds/Workspace/DiT"
CSV_PATH = "/home/ds/Workspace/moyi/exp-std-csv/train_top10_aug_sym.csv"
OUT_DIR = "/home/ds/Workspace/moyi/data/top10_style23/shards_img_aug"
VAE_PATH = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
BATCH_SIZE = 256
SHARD_SIZE = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)


class AugmentedImageDataset(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p = os.path.join(BASE_DIR, r["image_path"])
        img_id = str(r.get("img_id", f"{idx:06d}"))

        try:
            im = Image.open(p).convert("RGB")
            if im.size != (256, 256):
                im = im.resize((256, 256), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 256, 256)
            return tensor, img_id, True
        except Exception as e:
            dummy = torch.zeros(3, 256, 256, dtype=torch.float32)
            return dummy, img_id, False


def main():
    print("=" * 80)
    print("【开始批量 VAE Latent 编码 (77,823 样本)】")
    print(f"  CSV 路径: {CSV_PATH}")
    print(f"  输出目录: {OUT_DIR}")
    print(f"  VAE 路径: {VAE_PATH}")
    print(f"  Device  : {DEVICE}")
    print("=" * 80)

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    total_rows = len(rows)
    print(f"总计载入记录: {total_rows:,} 条")

    # 检查已存在的分片
    existing_shards = sorted(glob.glob(os.path.join(OUT_DIR, "shard_*.npz")))
    already_encoded = 0
    start_shard_idx = 0

    if existing_shards:
        # 验证最后一个分片是否完整
        last_shard = existing_shards[-1]
        data = np.load(last_shard)
        last_len = len(data["img_ids"])
        if last_len == SHARD_SIZE:
            start_shard_idx = len(existing_shards)
            already_encoded = start_shard_idx * SHARD_SIZE
        else:
            # 最后一个分片不完整，重新编它
            start_shard_idx = len(existing_shards) - 1
            already_encoded = start_shard_idx * SHARD_SIZE
            os.remove(last_shard)
        print(
            f"检测到断点: 已存在 {start_shard_idx} 个完整分片 ({already_encoded:,} 条)，从第 {already_encoded:,} 条继续编码！"
        )

    if already_encoded >= total_rows:
        print(f"全部分片已完整生成 (共 {len(existing_shards)} 个)，直接完成！")
        return

    remaining_rows = rows[already_encoded:]
    dataset = AugmentedImageDataset(remaining_rows)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print(f"载入 VAE 模型从: {VAE_PATH} ...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE).eval()
    scaling_factor = 0.18215

    shard_idx = start_shard_idx
    cur_latents = []
    cur_ids = []
    total_new = 0

    t0 = time.time()

    with torch.no_grad():
        for batch_imgs, batch_ids, batch_mask in tqdm(
            loader, desc="VAE Encoding"
        ):
            batch_imgs = batch_imgs.to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lat = (
                    vae.encode(batch_imgs).latent_dist.mode() * scaling_factor
                )

            lat_cpu = lat.half().cpu().numpy()
            for l_sample, i_id, valid in zip(
                lat_cpu, batch_ids, batch_mask.numpy()
            ):
                if valid:
                    cur_latents.append(l_sample)
                    cur_ids.append(str(i_id))
                    total_new += 1

                if len(cur_latents) >= SHARD_SIZE:
                    shard_file = os.path.join(
                        OUT_DIR, f"shard_{shard_idx:05d}.npz"
                    )
                    np.savez(
                        shard_file,
                        latents=np.array(cur_latents, dtype=np.float16),
                        img_ids=np.array(cur_ids, dtype=object),
                    )
                    shard_idx += 1
                    cur_latents = []
                    cur_ids = []

        if len(cur_latents) > 0:
            shard_file = os.path.join(OUT_DIR, f"shard_{shard_idx:05d}.npz")
            np.savez(
                shard_file,
                latents=np.array(cur_latents, dtype=np.float16),
                img_ids=np.array(cur_ids, dtype=object),
            )
            shard_idx += 1

    dt = time.time() - t0
    print("=" * 80)
    print(
        f"VAE Encoding 全部完成！本次新增: {total_new:,} 条，耗时: {dt:.1f} 秒 ({total_new/dt:.1f} 图/秒)"
    )
    print(f"总计分片数: {shard_idx} 个分片 (.npz)")
    print(f"产物目录  : {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
