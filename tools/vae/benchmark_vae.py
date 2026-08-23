# -*- coding: utf-8 -*-
"""
benchmark_vae.py — 全面对比 4 个 VAE 的重建质量.

测试项:
  1. MSE (像素重建)
  2. SSIM (结构相似, 批量向量化)
  3. LPIPS (感知距离, 如有)
  4. latent 统计 (mean/std/min/max/shape)
  5. 保存可视化对比图 (原图 + 各 VAE 重建)

VAE:
  A. sd-vae-ft-ema      (f8, 4ch, 83.7M, 3ch IO)
  B. sd-vae-ft-ema-gray (f8, 4ch, 83.6M, 1ch IO)
  C. kl-f4              (f4, 3ch, 55.3M, 3ch IO)
  D. kl-f4-gray         (f4, 3ch, 55.3M, 1ch IO)

用法:
  python tools/vae/benchmark_vae.py --csv 5script/train_top6.csv --n 200 --device cpu --out tools/vae/bench_results
"""
import os
import sys
import csv
import json
import time
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# ---------- SSIM (batch, vectorized) ----------
def _make_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    g = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    win = g.reshape(1, 1, window_size, 1) @ g.reshape(1, 1, 1, window_size)
    return win.expand(channels, 1, window_size, window_size).contiguous().to(device)


def batch_ssim(x, y, data_range=2.0, device="cpu"):
    """x, y: (B, C, H, W) in [-1,1]. Returns list of per-image SSIM."""
    B, C, H, W = x.shape
    win = _make_window(11, 1.5, C, device)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = 5
    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = F.conv2d(x ** 2, win, padding=pad, groups=C) - mu_x2
    sy2 = F.conv2d(y ** 2, win, padding=pad, groups=C) - mu_y2
    sxy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    # per-image mean
    return ssim_map.mean(dim=[1, 2, 3]).tolist()


# ---------- LPIPS ----------
_lpips_fn = None
def batch_lpips(x, y, device):
    """x, y: (B, C, H, W) in [-1,1]. Returns list of LPIPS values."""
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
            _lpips_fn.training = False
        except ImportError:
            return None
    with torch.no_grad():
        # LPIPS expects [-1,1]
        vals = _lpips_fn(x, y).squeeze(-1).squeeze(-1).tolist()
    return vals


# ---------- Data ----------
def load_images(csv_path, n=200, size=256):
    """Load n grayscale images, return (N, 1, H, W) in [-1,1]."""
    imgs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(imgs) >= n:
                break
            rel = row.get("image_path", "")
            path = rel if os.path.isabs(rel) else rel
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("L").resize((size, size))
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0
            imgs.append(torch.from_numpy(arr).unsqueeze(0))
    if not imgs:
        return None
    print(f"  loaded {len(imgs)} images")
    return torch.stack(imgs)


# ---------- Eval ----------
@torch.no_grad()
def eval_vae(vae, images_gray, device, batch=4, is_1ch=True, save_dir=None, save_n=8,
             skip_lpips=False):
    """
    images_gray: (N, 1, H, W) in [-1,1]
    Returns dict: mse_mean, mse_std, ssim_mean, ssim_std, lpips_mean, latent_stats, recon_samples
    """
    all_mse = []
    all_ssim = []
    all_lpips = []
    latent_vals = []

    recon_samples = None  # save first batch for visualization
    saved = 0

    # LPIPS is optional — only init if available, skip otherwise
    use_lpips = not skip_lpips

    for i in range(0, len(images_gray), batch):
        x_gray = images_gray[i:i+batch].to(device)

        if is_1ch:
            x_in = x_gray
        else:
            x_in = x_gray.repeat(1, 3, 1, 1)

        posterior = vae.encode(x_in).latent_dist
        z = posterior.sample()
        dec = vae.decode(z / vae.config.scaling_factor).sample

        if is_1ch:
            dec_gray = dec
        else:
            dec_gray = dec.mean(dim=1, keepdim=True)

        # MSE per image
        per_mse = ((dec_gray - x_gray) ** 2).mean(dim=[1, 2, 3]).tolist()
        all_mse.extend(per_mse)

        # SSIM per image
        ssim_vals = batch_ssim(x_gray, dec_gray, device=device)
        all_ssim.extend(ssim_vals)

        # LPIPS
        if not skip_lpips:
            lp = batch_lpips(x_gray, dec_gray, device)
            if lp is not None:
                all_lpips.extend(lp)
            else:
                skip_lpips = True  # init failed, stop trying

        # latent stats (first batch)
        if i == 0:
            latent_vals = z.detach().cpu()
            recon_samples = {
                "orig": x_gray[:save_n].cpu(),
                "recon": dec_gray[:save_n].cpu(),
            }

        if (i // batch) % 10 == 0:
            print(f"    batch {i//batch}/{(len(images_gray)+batch-1)//batch} done", flush=True)

    return {
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "lpips_mean": float(np.mean(all_lpips)) if all_lpips else None,
        "lpips_std": float(np.std(all_lpips)) if all_lpips else None,
        "latent_shape": list(latent_vals.shape[1:]) if latent_vals is not None else None,
        "latent_size": int(latent_vals.numel()) // len(images_gray) if latent_vals is not None else None,
        "latent_mean": float(latent_vals.mean().item()) if latent_vals is not None else None,
        "latent_std": float(latent_vals.std().item()) if latent_vals is not None else None,
        "latent_min": float(latent_vals.min().item()) if latent_vals is not None else None,
        "latent_max": float(latent_vals.max().item()) if latent_vals is not None else None,
        "recon_samples": recon_samples,
        "n_images": len(all_mse),
    }


def save_comparison_grid(results, out_dir):
    """Save a grid of original vs recon for each VAE."""
    import math
    n_vaes = len(results)
    n_samples = min(8, list(results.values())[0]["recon_samples"]["orig"].shape[0])

    # Layout: rows = n_samples, cols = 1 (orig) + n_vaes (recon)
    cols = 1 + n_vaes
    fig_w = cols * 2
    fig_h = n_samples * 2

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping grid")
        return

    fig, axes = plt.subplots(n_samples, cols, figsize=(fig_w, fig_h))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_samples):
        # Original
        axes[row, 0].imshow(results[list(results.keys())[0]]["recon_samples"]["orig"][row, 0],
                            cmap="gray", vmin=-1, vmax=1)
        axes[row, 0].set_title("Original" if row == 0 else "", fontsize=8)
        axes[row, 0].axis("off")
        # Recons
        for col, (name, r) in enumerate(results.items()):
            axes[row, col + 1].imshow(r["recon_samples"]["recon"][row, 0],
                                      cmap="gray", vmin=-1, vmax=1)
            axes[row, col + 1].set_title(name if row == 0 else "", fontsize=7)
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "vae_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_top6.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="tools/vae/bench_results")
    ap.add_argument("--save-grid", action="store_true")
    ap.add_argument("--skip-lpips", action="store_true")
    ap.add_argument("--vaes", nargs="*",
                    default=["pretrained_models/sd-vae-ft-ema",
                             "pretrained_models/sd-vae-ft-ema-gray",
                             "pretrained_models/kl-f4",
                             "pretrained_models/kl-f4-gray"])
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.n} grayscale images from {args.csv}...")
    images = load_images(args.csv, args.n)
    if images is None:
        print("ERROR: no images found")
        return
    print(f"  shape: {images.shape}")

    from diffusers import AutoencoderKL

    results = {}
    for vae_path in args.vaes:
        if not os.path.exists(vae_path):
            print(f"\n[SKIP] {vae_path} not found")
            continue

        name = os.path.basename(vae_path)
        is_1ch = "gray" in name or vae_path.endswith("-gray")
        print(f"\n{'='*50}")
        print(f"[{name}] {'1ch' if is_1ch else '3ch'} IO")
        print(f"{'='*50}")

        t0 = time.time()
        vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
        n_params = sum(p.numel() for p in vae.parameters())
        print(f"  Params: {n_params/1e6:.1f}M")

        r = eval_vae(vae, images, device, args.batch, is_1ch, save_n=8,
                    skip_lpips=args.skip_lpips)
        r["params_M"] = n_params / 1e6
        r["name"] = name
        r["is_1ch"] = is_1ch
        r["time_sec"] = time.time() - t0
        results[name] = r

        print(f"  MSE:   {r['mse_mean']:.6f} ± {r['mse_std']:.6f}")
        print(f"  SSIM:  {r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
        if r["lpips_mean"] is not None:
            print(f"  LPIPS: {r['lpips_mean']:.4f} ± {r['lpips_std']:.4f}")
        print(f"  Latent: {r['latent_shape']} ({r['latent_size']} vals/img)")
        print(f"  Latent stats: mean={r['latent_mean']:.4f} std={r['latent_std']:.4f} "
              f"min={r['latent_min']:.2f} max={r['latent_max']:.2f}")
        print(f"  Time: {r['time_sec']:.1f}s")

        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'VAE':<25} {'IO':>4} {'Params':>7} {'MSE':>10} {'SSIM':>10} {'Latent':>12} {'Time':>8}")
    print(f"{'-'*90}")
    for name, r in results.items():
        io = "1ch" if r["is_1ch"] else "3ch"
        lp = f"{r['lpips_mean']:.4f}" if r["lpips_mean"] is not None else "N/A"
        lat = f"{r['latent_shape']}" if r["latent_shape"] else "N/A"
        print(f"{name:<25} {io:>4} {r['params_M']:>6.1f}M {r['mse_mean']:>10.6f} "
              f"{r['ssim_mean']:>10.4f} {lat:>12} {r['time_sec']:>7.1f}s")
    print(f"{'='*90}")

    # Save JSON
    json_path = os.path.join(args.out, "results.json")
    json_data = {}
    for name, r in results.items():
        jr = {k: v for k, v in r.items() if k != "recon_samples"}
        json_data[name] = jr
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved {json_path}")

    # Save comparison grid
    if args.save_grid:
        save_comparison_grid(results, args.out)


if __name__ == "__main__":
    main()
