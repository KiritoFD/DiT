# -*- coding: utf-8 -*-
"""
estimate_scaling_factor.py — 估计 VAE latent 的 scaling factor.

scaling_factor = 1 / std(latent_samples)  (使 latent std ≈ 1, 利于 diffusion 训练)

用法:
  python tools/vae/estimate_scaling_factor.py --vae pretrained_models/kl-f4 --csv 5script/train_top6.csv --n 500
  python tools/vae/estimate_scaling_factor.py --vae pretrained_models/sd-vae-ft-ema --csv 5script/train_top6.csv --n 500
"""
import os
import sys
import csv
import re
import argparse
import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    in_ch = vae.config.in_channels
    print(f"VAE: in_ch={in_ch}, latent_ch={vae.config.latent_channels}")

    # Load images
    imgs = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(imgs) >= args.n:
                break
            rel = row.get("image_path", "")
            path = rel if os.path.isabs(rel) else rel
            if not os.path.exists(path):
                continue
            img = Image.open(path)
            if in_ch == 1:
                img = img.convert("L")
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                img = img.convert("RGB")
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
            imgs.append(t)
    print(f"Loaded {len(imgs)} images")

    # Encode, collect stats
    all_latents = []
    with torch.no_grad():
        for i in range(0, len(imgs), args.batch):
            batch = torch.stack(imgs[i:i+args.batch]).to(device)
            z = vae.encode(batch).latent_dist.sample()
            all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents, dim=0)
    print(f"\nLatent shape: {all_latents.shape}")
    print(f"  mean: {all_latents.mean().item():.4f}")
    print(f"  std:  {all_latents.std().item():.4f}")
    print(f"  min:  {all_latents.min().item():.4f}")
    print(f"  max:  {all_latents.max().item():.4f}")
    print(f"\nRecommended scaling_factor = 1/std = {1.0 / all_latents.std().item():.6f}")
    print(f"  (sd-vae uses 0.18215, which corresponds to std≈5.49)")


if __name__ == "__main__":
    main()
