# -*- coding: utf-8 -*-
"""
eval_vae.py — 比较不同 VAE 的重建质量 (MSE/SSIM/LPIPS).

比较对象:
  1. sd-vae-ft-ema (f8, 4ch, 当前用)
  2. kl-f4 (f4, 3ch, 新转换)
  3. (可选) 微调后的 VAE

指标:
  - MSE (像素)
  - SSIM (结构相似)
  - latent MSE (encode→decode 往返)

用法:
  python tools/vae/eval_vae.py --vae1 pretrained_models/sd-vae-ft-ema --vae2 pretrained_models/kl-f4 --data 5script/eval100_top30.csv --n 100
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Add src for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def load_vae(path, device="cpu"):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(path).to(device).eval()


def load_images(csv_path, n=100, img_root="final_images"):
    """Load n images from CSV, return tensor (N, 3, 256, 256) in [-1, 1]."""
    import csv
    imgs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            img_id = row.get("img_id") or row.get("id") or str(i)
            path = os.path.join(img_root, f"{img_id}.png")
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB").resize((256, 256))
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    if not imgs:
        print(f"WARNING: no images found in {csv_path}")
        return None
    return torch.stack(imgs)


@torch.no_grad()
def eval_recon(vae, images, device, batch=8):
    """Encode → decode → MSE/SSIM."""
    import torch.nn.functional as Fn

    # SSIM
    def _gaussian_window(window_size=11, sigma=1.5):
        g = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return (g.reshape(1, 1, window_size, 1) @ g.reshape(1, 1, 1, window_size)).to(device)

    win = _gaussian_window(11, 1.5)

    def _ssim(x, y, data_range=2.0, window_size=11):
        if x.shape[1] == 3:
            return sum(_ssim(x[:, i:i+1], y[:, i:i+1], data_range, window_size) for i in range(3)) / 3
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

    mse_sum = 0.0
    ssim_sum = 0.0
    cnt = 0
    latent_shapes = []

    for i in range(0, len(images), batch):
        x = images[i:i+batch].to(device)
        # Encode
        posterior = vae.encode(x).latent_dist
        z = posterior.sample()
        # Decode
        dec = vae.decode(z / vae.config.scaling_factor).sample

        mse_sum += F.mse_loss(dec, x).item() * x.shape[0]
        for k in range(x.shape[0]):
            ssim_sum += _ssim(x[k:k+1], dec[k:k+1])
            cnt += 1
        if i == 0:
            latent_shapes.append(z.shape)

    return mse_sum / cnt, ssim_sum / cnt, latent_shapes[0] if latent_shapes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae1", default="pretrained_models/sd-vae-ft-ema", help="VAE 1 (baseline)")
    ap.add_argument("--vae2", default="pretrained_models/kl-f4", help="VAE 2 (comparison)")
    ap.add_argument("--data", default="5script/eval100_top30.csv", help="eval CSV")
    ap.add_argument("--img-root", default="final_images", help="image root")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load images
    print(f"Loading {args.n} images from {args.data}...")
    images = load_images(args.data, args.n, args.img_root)
    if images is None:
        return
    print(f"  Loaded {len(images)} images, shape={images.shape}")

    for name, path in [("vae1", args.vae1), ("vae2", args.vae2)]:
        if not os.path.exists(path):
            print(f"\n[{name}] {path} not found, skipping")
            continue
        print(f"\n[{name}] Loading {path}...")
        vae = load_vae(path, device)
        n_params = sum(p.numel() for p in vae.parameters())
        print(f"  Params: {n_params/1e6:.1f}M")
        print(f"  Config: in={vae.config.in_channels}, latent={vae.config.latent_channels}, "
              f"blocks={vae.config.block_out_channels}")

        mse, ssim, z_shape = eval_recon(vae, images, device, args.batch)
        print(f"  Results: MSE={mse:.6f}, SSIM={ssim:.4f}")
        print(f"  Latent shape (per image): {z_shape if z_shape else 'N/A'}")
        print(f"  Latent size: {z_shape[1]*z_shape[2]*z_shape[3] if z_shape else 'N/A'} values")

        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
