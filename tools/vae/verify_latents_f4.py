# -*- coding: utf-8 -*-
"""
verify_latents_f4.py — encode 完后验证:
  1. 从全量 shard 统计 latent mean/std/min/max → 确认 scaling_factor
  2. 随机挑 N 张: decode(latent) vs GT → MSE/SSIM 底噪

用法 (远程):
  python tools/vae/verify_latents_f4.py --shards final_latents_f4 --vae pretrained_models/kl-f4 --img-root final_images --n 100
"""
import os, sys, glob, random, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    g = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.reshape(1, 1, window_size, 1) @ g.reshape(1, 1, 1, window_size))


def _ssim(x, y, data_range=2.0, win=None):
    """SSIM, handles multi-channel by computing per-channel and averaging."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    if x.shape[1] > 1:
        return sum(_ssim(x[:, i:i+1], y[:, i:i+1], data_range, win) for i in range(x.shape[1])) / x.shape[1]
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="final_latents_f4")
    ap.add_argument("--vae", default="pretrained_models/kl-f4")
    ap.add_argument("--img-root", default="final_images")
    ap.add_argument("--n", type=int, default=100, help="N images for recon test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scaling-factor", type=float, default=0.102079,
                    help="scaling factor used during encoding (to un-scale for decode)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    shards = sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))
    if not shards:
        print(f"ERROR: no shards in {args.shards}")
        return
    print(f"Found {len(shards)} shards", flush=True)

    # === 1. 统计全量 latent 分布 (分 shard 累积, 用 float64 避免溢出) ===
    print(f"\n{'='*60}")
    print("Step 1: Computing latent statistics from all shards...")
    print(f"{'='*60}")
    all_mean_sum = 0.0
    all_sq_sum = 0.0
    all_min = float('inf')
    all_max = float('-inf')
    total_count = 0

    for sp in shards:
        d = np.load(sp)
        lat = d["latents"]  # (N, 3, 64, 64) fp16
        lat_f32 = lat.astype(np.float32)
        all_mean_sum += float(lat_f32.sum(dtype=np.float64))
        all_sq_sum += float((lat_f32 * lat_f32).sum(dtype=np.float64))
        all_min = min(all_min, float(lat_f32.min()))
        all_max = max(all_max, float(lat_f32.max()))
        total_count += lat.size
        d.close()

    global_mean = all_mean_sum / total_count
    global_var = all_sq_sum / total_count - global_mean ** 2
    global_std = global_var ** 0.5
    recommended_scaling = 1.0 / global_std

    print(f"  Total values: {total_count:,}")
    print(f"  Mean:   {global_mean:.6f}")
    print(f"  Std:    {global_std:.6f}")
    print(f"  Min:    {all_min:.4f}")
    print(f"  Max:    {all_max:.4f}")
    print(f"  Recommended scaling_factor = 1/std = {recommended_scaling:.6f}")
    if recommended_scaling > 0:
        print(f"  (current config uses 0.102079, diff = {abs(recommended_scaling - 0.102079)/recommended_scaling*100:.2f}%)")
    else:
        print(f"  WARNING: std=0 or inf, statistics computation failed")

    # === 2. 随机挑 N 张做 reconstruct 底噪测试 ===
    print(f"\n{'='*60}")
    print(f"Step 2: Reconstruction test ({args.n} random images)")
    print(f"{'='*60}")

    # Collect all img_ids from shards
    all_ids = []
    for sp in shards:
        d = np.load(sp)
        all_ids.extend(d["img_ids"].tolist())
        d.close()
    print(f"  Total img_ids: {len(all_ids)}")

    # Sample N
    random.seed(42)
    sample_ids = random.sample(all_ids, min(args.n, len(all_ids)))

    # Load VAE
    from diffusers.models import AutoencoderKL
    print(f"  Loading VAE: {args.vae}")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()

    # Build id→shard lookup
    id_to_shard = {}
    for sp in shards:
        d = np.load(sp)
        for j, iid in enumerate(d["img_ids"]):
            id_to_shard[int(iid)] = (sp, j)
        d.close()

    win = _gaussian_window(11, 1.5, device)
    mse_list = []
    ssim_list = []
    t0 = time.time()

    for i in range(0, len(sample_ids), args.batch):
        batch_ids = sample_ids[i:i + args.batch]
        # Load latents from shards
        latents = []
        gts = []
        for iid in batch_ids:
            sp, j = id_to_shard[iid]
            d = np.load(sp)
            lat = torch.from_numpy(np.array(d["latents"][j], copy=True)).float()  # (3, 64, 64)
            d.close()
            latents.append(lat)

            # Load GT image
            img_path = os.path.join(args.img_root, f"{iid}.png")
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            gts.append(torch.from_numpy(arr).permute(2, 0, 1))

        z = torch.stack(latents).to(device)  # already scaled (encoded with scaling_factor)
        gt = torch.stack(gts).to(device)

        with torch.no_grad():
            # Un-scale: latent was stored as encode(x).sample() * scaling_factor
            # To decode: decode(latent / scaling_factor)
            dec = vae.decode(z / args.scaling_factor).sample  # (B, 3, 256, 256)

        # MSE
        per_mse = ((dec - gt) ** 2).mean(dim=[1, 2, 3]).tolist()
        mse_list.extend(per_mse)

        # SSIM per image
        for k in range(dec.shape[0]):
            ssim_val = _ssim(gt[k:k+1], dec[k:k+1], data_range=2.0, win=win)
            ssim_list.append(ssim_val)

        if (i // args.batch) % 5 == 0:
            print(f"    batch {i//args.batch}/{(len(sample_ids)+args.batch-1)//args.batch}", flush=True)

    elapsed = time.time() - t0
    mse_mean = float(np.mean(mse_list))
    mse_std = float(np.std(mse_list))
    ssim_mean = float(np.mean(ssim_list))
    ssim_std = float(np.std(ssim_list))

    print(f"\n  Results ({len(mse_list)} images, {elapsed:.0f}s):")
    print(f"  MSE:  {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Latent shape:  (3, 64, 64) = 12288 values/img")
    print(f"  Latent mean:   {global_mean:.6f}")
    print(f"  Latent std:    {global_std:.6f}")
    print(f"  Scaling factor (1/std): {recommended_scaling:.6f}")
    print(f"  Recon MSE:     {mse_mean:.6f}")
    print(f"  Recon SSIM:    {ssim_mean:.4f}")
    print(f"  (sd-vae-ft-ema for reference: MSE=0.206, SSIM=0.723)")
    print(f"  (kl-f4 benchmark:           MSE=0.074, SSIM=0.870)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
