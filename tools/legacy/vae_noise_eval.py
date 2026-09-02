#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VAE 本底噪声监测 (eval集, CPU, 快速版)。
对 eval100_top30_clean.csv 的 79 张图, 用 2 个 VAE × 2 patch 组合
encode→decode, 统计 MSE/SSIM (用实际 scale 参数)。
patch 不影响 VAE 重建 (只影响 DiT token 化), 跑全 4 组验证这一点。
"""
import os, sys, csv, json, time, gc
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_num_threads(32)

BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(BASE, "5script", "eval100_top30_clean.csv")
IMG_ROOT = BASE
OUT_DIR = os.path.join(BASE, "tools", "vae_noise_results")
os.makedirs(OUT_DIR, exist_ok=True)

VAE_CONFIGS = [
    # name, vae_path, downscale, latent_ch, scale, patch
    ("kl-f4_p4", os.path.join(BASE, "pretrained_models", "kl-f4"),
     4, 3, 0.102079, 4),
    ("kl-f4_p2", os.path.join(BASE, "pretrained_models", "kl-f4"),
     4, 3, 0.102079, 2),
    ("sd-vae-ft-ema_p4", os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"),
     8, 4, 0.18215, 4),
    ("sd-vae-ft-ema_p2", os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"),
     8, 4, 0.18215, 2),
]


def load_paths():
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


def compute_ssim(img1, img2, win=7):
    c1, c2 = 0.01**2, 0.03**2
    ssims = []
    for ch in range(3):
        x = img1[:, :, ch].astype(np.float64)
        y = img2[:, :, ch].astype(np.float64)
        mu_x = uniform_filter(x, size=win)
        mu_y = uniform_filter(y, size=win)
        mu_x2 = mu_x**2; mu_y2 = mu_y**2; mu_xy = mu_x * mu_y
        sigma_x2 = uniform_filter(x*x, size=win) - mu_x2
        sigma_y2 = uniform_filter(y*y, size=win) - mu_y2
        sigma_xy = uniform_filter(x*y, size=win) - mu_xy
        ssim_map = ((2*mu_xy+c1)*(2*sigma_xy+c2)) / ((mu_x2+mu_y2+c1)*(sigma_x2+sigma_y2+c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))


def evaluate(name, vae_path, ds, ch, scale, patch, paths):
    from diffusers.models import AutoencoderKL
    print(f"\n{'='*60}")
    print(f"=== {name} (ds={ds}, ch={ch}, scale={scale}, patch={patch}) ===")
    device = "cpu"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256)
        lat = vae.encode(dummy).latent_dist.sample()
        lat_shape = tuple(lat.shape[1:])
        h, w = lat_shape[1], lat_shape[2]
        tokens = (h//2)*(w//2) if patch == 2 else (h//4)*(w//4)
    print(f"  latent: {lat_shape}, patch{patch} → {tokens} tokens, images: {len(paths)}")
    print(f"  scale={scale} (实际使用)")

    all_mse = []
    all_ssim = []
    t0 = time.time()
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            latent = vae.encode(x).latent_dist.sample()
            decoded = vae.decode(latent).sample.clamp(0, 1).cpu().numpy()[0]
            orig = arr
            rec = decoded.transpose(1, 2, 0)
            mse = float(np.mean((orig - rec)**2))
            ssim = compute_ssim(orig, rec)
            all_mse.append(mse)
            all_ssim.append(ssim)

    elapsed = time.time() - t0
    all_mse = np.array(all_mse)
    all_ssim = np.array(all_ssim)
    summary = {
        "name": name, "vae_path": vae_path,
        "downscale": ds, "latent_channels": ch,
        "scaling_factor": scale, "patch": patch,
        "latent_shape": lat_shape, "tokens": tokens,
        "n_images": len(all_mse),
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "mse_median": float(np.median(all_mse)),
        "mse_max": float(np.max(all_mse)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "ssim_median": float(np.median(all_ssim)),
        "ssim_min": float(np.min(all_ssim)),
        "elapsed_sec": elapsed,
    }
    print(f"  DONE: {len(all_mse)} imgs in {elapsed:.1f}s")
    print(f"  MSE:  mean={summary['mse_mean']:.6f} std={summary['mse_std']:.6f} "
          f"median={summary['mse_median']:.6f} max={summary['mse_max']:.6f}")
    print(f"  SSIM: mean={summary['ssim_mean']:.4f} std={summary['ssim_std']:.4f} "
          f"median={summary['ssim_median']:.4f} min={summary['ssim_min']:.4f}")
    del vae
    gc.collect()
    return summary


def main():
    paths = load_paths()
    print(f"=== VAE 本底噪声监测 (eval集, {len(paths)} 张, CPU) ===")
    print(f"CSV: {CSV_PATH}")
    print(f"配置数: {len(VAE_CONFIGS)} (2 VAE × 2 patch)")

    summaries = []
    for name, vp, ds, ch, sc, pt in VAE_CONFIGS:
        s = evaluate(name, vp, ds, ch, sc, pt, paths)
        summaries.append(s)

    print(f"\n{'='*60}")
    print(f"=== 汇总 ({len(summaries)} 配置, {len(paths)} 张图) ===")
    print(f"{'配置':<22} {'VAE':<6} {'ds':<4} {'patch':<6} {'tokens':<8} "
          f"{'scale':<10} {'MSE':<12} {'SSIM':<10} {'最差SSIM':<10}")
    for s in summaries:
        vn = "kl-f4" if s["downscale"] == 4 else "sd-vae"
        print(f"{s['name']:<22} {vn:<6} {s['downscale']:<4} {s['patch']:<6} {s['tokens']:<8} "
              f"{s['scaling_factor']:<10} {s['mse_mean']:.6f}    {s['ssim_mean']:.4f}    {s['ssim_min']:.4f}")
    with open(os.path.join(OUT_DIR, "vae_noise_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {OUT_DIR}/vae_noise_eval_summary.json")


if __name__ == "__main__":
    main()
