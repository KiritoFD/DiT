# -*- coding: utf-8 -*-
"""encode_reprocessed_shards.py — 将重新处理的高清无断笔书法样本编码为 VAE Latents 与 Skel 骨架分片。

设计功能:
  1. 读取 assets/reprocessed_hcsu_all.csv (包含 wild / bei / tie 22,638 条样本)
  2. 使用 SD-VAE 将高清二值图编码为 latent (N, 4, 32, 32) float16
  3. 可选同步生成骨架 (aux_skel3)，保证监督骨架同样无断笔
  4. 产物分片保存至 data/hcsu_reprocessed_clean/shards_img/，并准备好替换对应 shard
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.morphology import skeletonize
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"
SHARD_SIZE = 4096
ST = generate_binary_structure(2, 2)


def get_transforms(size=256):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/reprocessed_hcsu_all.csv")
    ap.add_argument("--out-img-shards", default="data/hcsu_reprocessed_clean/shards_img")
    ap.add_argument("--out-skel-shards", default="data/hcsu_reprocessed_clean/shards_aux_skel3")
    ap.add_argument("--vae-path", default=VAE_PATH)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--with-skel", action="store_true", default=True)
    args = ap.parse_args()

    print(f"=== VAE Latent Encoding Pipeline for Reprocessed Calligraphy ===")
    print(f"Reading CSV: {args.csv}")
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"Total samples: {len(rows)}")

    valid_rows = [r for r in rows if r.get("status") == "ok" and os.path.exists(r.get("new_image_path", ""))]
    print(f"Valid samples with existing images: {len(valid_rows)}")

    os.makedirs(args.out_img_shards, exist_ok=True)
    if args.with_skel:
        os.makedirs(args.out_skel_shards, exist_ok=True)

    print(f"Loading VAE from {args.vae_path} on {args.device}...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(args.device)
    vae.eval()

    tf = get_transforms(256)
    n = len(valid_rows)
    total_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE

    print(f"Encoding {n} samples into {total_shards} shards (shard_size={SHARD_SIZE})...")
    t0 = time.time()

    for s_idx in range(total_shards):
        s_start = s_idx * SHARD_SIZE
        s_end = min(s_start + SHARD_SIZE, n)
        shard_rows = valid_rows[s_start:s_end]

        latents_list = []
        skels_list = []
        ids_list = []

        for b_start in range(0, len(shard_rows), args.batch_size):
            b_end = min(b_start + args.batch_size, len(shard_rows))
            batch_slice = shard_rows[b_start:b_end]

            imgs_tensor = []
            for r in batch_slice:
                p = r["new_image_path"]
                im = Image.open(p).convert("RGB")
                imgs_tensor.append(tf(im))

                if args.with_skel:
                    # Target skeleton: 1px skeleton dilate 1 iter -> 3px
                    gray = np.asarray(im.convert("L"))
                    sk = skeletonize(gray < 128)
                    sk_3px = binary_dilation(sk, ST, iterations=1)
                    sk_im = Image.fromarray(np.where(sk_3px, 0, 255).astype(np.uint8)).convert("RGB")
                    skels_list.append(tf(sk_im))

                ids_list.append(int(r["old_50k_id"]) if r.get("old_50k_id") else (s_start + len(ids_list)))

            batch_tensor = torch.stack(imgs_tensor).to(args.device)
            with torch.no_grad():
                latent_dist = vae.encode(batch_tensor).latent_dist
                lat = latent_dist.sample() * 0.18215
            latents_list.append(lat.half().cpu().numpy())

        # Save image latents shard
        all_latents = np.concatenate(latents_list, axis=0)
        img_shard_path = os.path.join(args.out_img_shards, f"shard_{s_idx:05d}.npz")
        np.savez_compressed(img_shard_path, latents=all_latents, img_ids=np.array(ids_list))
        print(f"Saved Shard {s_idx + 1}/{total_shards}: {img_shard_path} ({len(all_latents)} samples)")

        # Save skel latents shard if requested
        if args.with_skel:
            skel_batch = torch.stack(skels_list).to(args.device)
            skel_latents = []
            for b in range(0, len(skel_batch), args.batch_size):
                with torch.no_grad():
                    sk_lat = vae.encode(skel_batch[b:b+args.batch_size]).latent_dist.sample() * 0.18215
                skel_latents.append(sk_lat.half().cpu().numpy())
            all_skels = np.concatenate(skel_latents, axis=0)
            skel_shard_path = os.path.join(args.out_skel_shards, f"shard_{s_idx:05d}.npz")
            np.savez_compressed(skel_shard_path, latents=all_skels, img_ids=np.array(ids_list))

    print(f"\nAll shards successfully encoded in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
