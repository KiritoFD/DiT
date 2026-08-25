#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量 VAE 本底噪声监测 (GPU版, 优化)。
关键: patch 不影响 VAE 重建 (只影响 DiT token 化), 所以只跑 2 个 VAE:
  1. kl-f4 (downscale=4, latent_ch=3, scale=0.102079) → patch4=256 tokens, patch2=1024 tokens
  2. sd-vae-ft-ema (downscale=8, latent_ch=4, scale=0.18215) → patch4=64 tokens, patch2=256 tokens
全量 106,345 张, GPU encode→decode, 统计 MSE+SSIM (GPU SSIM)。
"""
import os, sys, csv, json, time, gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(BASE, "5script", "train_top30_clean.csv")
IMG_ROOT = BASE
OUT_DIR = os.path.join(BASE, "tools", "vae_noise_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"
BATCH_SIZE = 4   # GPU 共享 (s9 训练占 ~20G), 用极小 batch
N_LOADERS = 16   # 多线程读图 (IO 是瓶颈)

VAE_CONFIGS = [
    # name, vae_path, downscale, latent_ch, scale, patch(用于算 tokens)
    ("kl-f4_p4", os.path.join(BASE, "pretrained_models", "kl-f4"), 4, 3, 0.102079, 4),
    ("sd-vae-ft-ema_p4", os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"), 8, 4, 0.18215, 4),
]


def load_image_paths():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    paths = []
    for r in rows:
        p = r["image_path"]
        if p.startswith("final_images/"):
            p = p.replace("final_images/", "final_imgs_256/", 1)
        full = os.path.join(IMG_ROOT, p)
        if os.path.isfile(full):
            paths.append(full)
    return paths


def _load_one(p):
    img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_batch(paths):
    with ThreadPoolExecutor(max_workers=N_LOADERS) as pool:
        imgs = list(pool.map(_load_one, paths))
    arr = np.stack(imgs)  # (B,H,W,3)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # (B,3,H,W)


def gpu_ssim_batch(x, y, win=7):
    """Batch SSIM on GPU. x,y: (B,3,H,W) [0,1]"""
    C1, C2 = 0.01**2, 0.03**2
    pad = win // 2
    mu_x = F.avg_pool2d(x, win, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, win, stride=1, padding=pad)
    mu_x2 = mu_x**2; mu_y2 = mu_y**2; mu_xy = mu_x * mu_y
    sigma_x2 = F.avg_pool2d(x*x, win, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(y*y, win, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(x*y, win, stride=1, padding=pad) - mu_xy
    ssim_map = ((2*mu_xy+C1)*(2*sigma_xy+C2)) / ((mu_x2+mu_y2+C1)*(sigma_x2+sigma_y2+C2))
    return ssim_map.mean(dim=[1,2,3])  # (B,)


def evaluate_config(name, vae_path, downscale, latent_ch, scale, patch, paths):
    from diffusers.models import AutoencoderKL
    print(f"\n{'='*60}")
    print(f"=== {name} (ds={downscale}, ch={latent_ch}, scale={scale}, patch={patch}) ===")

    vae = AutoencoderKL.from_pretrained(vae_path).to(DEVICE).eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256, device=DEVICE)
        lat = vae.encode(dummy).latent_dist.sample()
        lat_shape = tuple(lat.shape[1:])
        h, w = lat_shape[1], lat_shape[2]
        tokens_p4 = (h//4)*(w//4)
        tokens_p2 = (h//2)*(w//2)
    print(f"  latent: {lat_shape}, patch4={tokens_p4} tokens, patch2={tokens_p2} tokens, images: {len(paths)}")

    # 预加载所有图片到内存 (106k x 256x256x3 float32 = ~75GB, 太大)
    # 改为: 预加载到 uint8 (106k x 256x256x3 = ~19GB), 用时转 float
    print(f"  preloading {len(paths)} images to RAM (uint8)...")
    t_load0 = time.time()
    all_uint8 = [None] * len(paths)
    def _preload(idx_path):
        idx, p = idx_path
        img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        all_uint8[idx] = np.asarray(img, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=N_LOADERS) as pool:
        list(pool.map(_preload, enumerate(paths)))
    load_time = time.time() - t_load0
    total_mb = sum(a.nbytes for a in all_uint8 if a is not None) / 1e6
    print(f"  preloaded in {load_time:.0f}s ({total_mb/1e3:.1f} GB)")

    all_mse = []
    all_ssim = []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch_imgs = all_uint8[i:i+BATCH_SIZE]
            if any(b is None for b in batch_imgs):
                n_done += len(batch_imgs)
                continue
            arr = np.stack([b.astype(np.float32)/255.0 for b in batch_imgs])
            x = torch.from_numpy(arr).permute(0,3,1,2).to(DEVICE)
            latent = vae.encode(x).latent_dist.sample()
            decoded = vae.decode(latent).sample.clamp(0, 1)
            mse = ((x - decoded)**2).mean(dim=[1,2,3])
            ssim = gpu_ssim_batch(x, decoded)
            all_mse.extend(mse.cpu().numpy().tolist())
            all_ssim.extend(ssim.cpu().numpy().tolist())
            n_done += len(batch_imgs)
            if n_done % 400 == 0:
                el = time.time() - t0
                rate = n_done / el
                eta = (len(paths) - n_done) / rate
                print(f"  [{name}] {n_done}/{len(paths)} ({rate:.1f} img/s, "
                      f"ETA {eta/60:.0f}min, MSE={np.mean(all_mse):.6f}, "
                      f"SSIM={np.mean(all_ssim):.4f})", flush=True)
            del x, latent, decoded, mse, ssim

    elapsed = time.time() - t0
    all_mse = np.array(all_mse)
    all_ssim = np.array(all_ssim)
    summary = {
        "name": name, "vae_path": vae_path,
        "downscale": downscale, "latent_channels": latent_ch,
        "scaling_factor": scale, "patch": patch,
        "latent_shape": lat_shape,
        "tokens_patch4": tokens_p4, "tokens_patch2": tokens_p2,
        "n_images": len(all_mse),
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "mse_median": float(np.median(all_mse)),
        "mse_p95": float(np.percentile(all_mse, 95)),
        "mse_max": float(np.max(all_mse)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "ssim_median": float(np.median(all_ssim)),
        "ssim_p5": float(np.percentile(all_ssim, 5)),
        "ssim_min": float(np.min(all_ssim)),
        "elapsed_sec": elapsed,
        "rate": len(all_mse) / elapsed,
    }
    print(f"\n  [{name}] DONE: {n_done} imgs in {elapsed:.0f}s ({n_done/elapsed:.1f} img/s)")
    print(f"  MSE:  mean={summary['mse_mean']:.6f} std={summary['mse_std']:.6f} "
          f"median={summary['mse_median']:.6f} p95={summary['mse_p95']:.6f} max={summary['mse_max']:.6f}")
    print(f"  SSIM: mean={summary['ssim_mean']:.4f} std={summary['ssim_std']:.4f} "
          f"median={summary['ssim_median']:.4f} p5={summary['ssim_p5']:.4f} min={summary['ssim_min']:.4f}")

    with open(os.path.join(OUT_DIR, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    del vae, all_uint8
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    paths = load_image_paths()
    print(f"=== VAE 本底噪声监测 (GPU优化版, {len(paths)} 张) ===")
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}, Loaders: {N_LOADERS}")
    print(f"配置数: {len(VAE_CONFIGS)} (patch不影响VAE重建, 每VAE只跑1次)")
    print(f"GPU mem free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    summaries = []
    for name, vae_path, ds, ch, scale, patch in VAE_CONFIGS:
        s = evaluate_config(name, vae_path, ds, ch, scale, patch, paths)
        summaries.append(s)

    print(f"\n{'='*60}")
    print(f"=== 汇总 ({len(summaries)} VAE, {len(paths)} 张图) ===")
    print(f"{'VAE':<20} {'ds':<4} {'ch':<4} {'scale':<10} "
          f"{'p4_tok':<7} {'p2_tok':<7} "
          f"{'MSE':<12} {'SSIM':<10} {'最差SSIM':<10}")
    for s in summaries:
        print(f"{s['name']:<20} {s['downscale']:<4} {s['latent_channels']:<4} "
              f"{s['scaling_factor']:<10} {s['tokens_patch4']:<7} {s['tokens_patch2']:<7} "
              f"{s['mse_mean']:.6f}    {s['ssim_mean']:.4f}    {s['ssim_min']:.4f}")
    with open(os.path.join(OUT_DIR, "vae_noise_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {OUT_DIR}/vae_noise_summary.json")


if __name__ == "__main__":
    main()
