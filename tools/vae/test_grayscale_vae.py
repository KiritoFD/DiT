# -*- coding: utf-8 -*-
"""
test_grayscale_vae.py — 本地测试黑白 VAE 的 encode/decode 质量.

对比:
  1. 原 3ch VAE: 灰度图复制3份 -> encode -> decode -> 取灰度
  2. 黑白 1ch VAE: 灰度图直接 -> encode -> decode
看谁的重建质量更好 (MSE/SSIM).
"""
import os
import sys
import csv
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def load_grayscale_images(csv_path, img_root, n=50, size=256):
    """Load n grayscale images, return (N, 1, 256, 256) in [-1, 1]."""
    imgs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(imgs) >= n:
                break
            img_id = row.get("img_id") or row.get("id")
            if img_id:
                path = os.path.join(img_root, f"{img_id}.png")
            else:
                # image_path column (relative path like "final_images/190.png")
                rel = row.get("image_path", "")
                path = rel  # relative to CWD
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("L").resize((size, size))
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            imgs.append(torch.from_numpy(arr).unsqueeze(0))  # [1, H, W]
    if not imgs:
        print(f"WARNING: no images found in {csv_path} / {img_root}")
        return None
    return torch.stack(imgs)


def gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    g = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.reshape(1, 1, window_size, 1) @ g.reshape(1, 1, 1, window_size)).to(device)


def ssim(x, y, data_range=2.0, win=None):
    """SSIM for single-channel images."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, win, padding=5)
    mu_y = F.conv2d(y, win, padding=5)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(x ** 2, win, padding=5) - mu_x2
    sy2 = F.conv2d(y ** 2, win, padding=5) - mu_y2
    sxy = F.conv2d(x * y, win, padding=5) - mu_xy
    m = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    return float(m.mean().item())


@torch.no_grad()
def eval_vae(vae, images_gray, device, batch=8, is_1ch=True):
    """
    images_gray: (N, 1, H, W) in [-1, 1]
    is_1ch: True for grayscale VAE, False for 3ch VAE (replicate input)
    """
    win = gaussian_window(device=device)
    mse_sum = 0.0
    ssim_sum = 0.0
    cnt = 0
    latent_total = 0

    for i in range(0, len(images_gray), batch):
        x_gray = images_gray[i:i+batch].to(device)  # (B, 1, H, W)
        if is_1ch:
            x_in = x_gray  # (B, 1, H, W)
        else:
            x_in = x_gray.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        # Encode
        posterior = vae.encode(x_in).latent_dist
        z = posterior.sample()
        latent_total += z.numel()

        # Decode
        dec = vae.decode(z / vae.config.scaling_factor).sample  # (B, C, H, W)

        if is_1ch:
            dec_gray = dec  # (B, 1, H, W)
        else:
            dec_gray = dec.mean(dim=1, keepdim=True)  # (B, 1, H, W) - average 3ch

        # Metrics
        mse_sum += F.mse_loss(dec_gray, x_gray).item() * x_gray.shape[0]
        for k in range(x_gray.shape[0]):
            ssim_sum += ssim(x_gray[k:k+1], dec_gray[k:k+1], win=win)
            cnt += 1

    return mse_sum / cnt, ssim_sum / cnt, latent_total / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae3", default="pretrained_models/sd-vae-ft-ema", help="3ch VAE (baseline)")
    ap.add_argument("--vae1", default="pretrained_models/sd-vae-ft-ema-gray", help="1ch grayscale VAE")
    ap.add_argument("--vae-klf4", default="pretrained_models/kl-f4", help="kl-f4 3ch VAE")
    ap.add_argument("--vae-klf4-gray", default="pretrained_models/kl-f4-gray", help="kl-f4 1ch VAE")
    ap.add_argument("--csv", default="5script/eval100_top30.csv")
    ap.add_argument("--img-root", default="final_images")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.n} grayscale images...")
    images = load_grayscale_images(args.csv, args.img_root, args.n)
    if images is None:
        return
    print(f"  {images.shape}")

    from diffusers import AutoencoderKL

    results = {}

    # 1. 原 3ch VAE (baseline)
    if os.path.exists(args.vae3):
        print(f"\n[3ch VAE] {args.vae3}")
        vae = AutoencoderKL.from_pretrained(args.vae3).to(device).eval()
        n_params = sum(p.numel() for p in vae.parameters())
        mse, ssim_val, lat_per_img = eval_vae(vae, images, device, args.batch, is_1ch=False)
        print(f"  Params: {n_params/1e6:.1f}M")
        print(f"  Latent per image: {lat_per_img} values")
        print(f"  MSE:  {mse:.6f}")
        print(f"  SSIM: {ssim_val:.4f}")
        results["3ch (original)"] = {"mse": mse, "ssim": ssim_val, "latent": lat_per_img}
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2. 黑白 1ch VAE
    if os.path.exists(args.vae1):
        print(f"\n[1ch VAE] {args.vae1}")
        vae = AutoencoderKL.from_pretrained(args.vae1).to(device).eval()
        n_params = sum(p.numel() for p in vae.parameters())
        mse, ssim_val, lat_per_img = eval_vae(vae, images, device, args.batch, is_1ch=True)
        print(f"  Params: {n_params/1e6:.1f}M")
        print(f"  Latent per image: {lat_per_img} values")
        print(f"  MSE:  {mse:.6f}")
        print(f"  SSIM: {ssim_val:.4f}")
        results["1ch (grayscale surgery)"] = {"mse": mse, "ssim": ssim_val, "latent": lat_per_img}
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3. kl-f4 3ch (if available)
    if os.path.exists(args.vae_klf4):
        print(f"\n[kl-f4 3ch] {args.vae_klf4}")
        vae = AutoencoderKL.from_pretrained(args.vae_klf4).to(device).eval()
        n_params = sum(p.numel() for p in vae.parameters())
        mse, ssim_val, lat_per_img = eval_vae(vae, images, device, args.batch, is_1ch=False)
        print(f"  Params: {n_params/1e6:.1f}M")
        print(f"  Latent per image: {lat_per_img} values")
        print(f"  MSE:  {mse:.6f}")
        print(f"  SSIM: {ssim_val:.4f}")
        results["kl-f4 3ch"] = {"mse": mse, "ssim": ssim_val, "latent": lat_per_img}
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. kl-f4 1ch (grayscale surgery)
    if os.path.exists(args.vae_klf4_gray):
        print(f"\n[kl-f4 1ch] {args.vae_klf4_gray}")
        vae = AutoencoderKL.from_pretrained(args.vae_klf4_gray).to(device).eval()
        n_params = sum(p.numel() for p in vae.parameters())
        mse, ssim_val, lat_per_img = eval_vae(vae, images, device, args.batch, is_1ch=True)
        print(f"  Params: {n_params/1e6:.1f}M")
        print(f"  Latent per image: {lat_per_img} values")
        print(f"  MSE:  {mse:.6f}")
        print(f"  SSIM: {ssim_val:.4f}")
        results["kl-f4 1ch (gray)"] = {"mse": mse, "ssim": ssim_val, "latent": lat_per_img}
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"{'VAE':<25} {'MSE':<12} {'SSIM':<10} {'Latent/img':<12}")
    print(f"{'-'*60}")
    for name, r in results.items():
        print(f"{name:<25} {r['mse']:<12.6f} {r['ssim']:<10.4f} {r['latent']:<12}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
