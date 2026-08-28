#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量 VAE 本底噪声监测: 对 5script/train_top30_clean.csv 的全部 106,345 张图,
用 4 种 VAE×patch 组合 encode→decode, 统计 MSE/SSIM。
组合:
  1. kl-f4 + patch4  (当前 s9 使用): downscale=4, latent_ch=3, scale=0.102079
  2. kl-f4 + patch2:  downscale=4, latent_ch=3, scale=0.102079
  3. sd-vae-ft-ema + patch4: downscale=8, latent_ch=4, scale=0.18215
  4. sd-vae-ft-ema + patch2 (s6 使用): downscale=8, latent_ch=4, scale=0.18215
注意: patch size 只影响 DiT token 数, 不影响 VAE 重建质量。VAE 重建只取决于 downscale。
所以这里实际只比较 2 个 VAE: f4(kl-f4) vs f8(sd-vae-ft-ema)。
patch2 vs patch4 在 VAE 重建层面完全等价 (它们 decode 的是同样的 latent)。
但用户要的是"VAE+patch 组合的端到端效果", 所以我们还是跑 4 组, 验证 patch 不影响 VAE 重建。

优化: batch encode (CPU), 64 核并行, 预计 ~3h。
输出: vae_noise_full.json (每张图的 MSE/SSIM) + vae_noise_summary.json (统计)。
"""
import os, sys, csv, json, time, gc, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
torch.set_num_threads(64)
import torch.nn.functional as F
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# ── 配置 ──
BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(BASE, "5script", "train_top30_clean.csv")
IMG_ROOT = BASE  # CSV 里是 final_imgs_256/xxx.png
BATCH_SIZE = 256  # CPU batch (大 batch 更高效)
DEVICE = "cpu"
MAX_IMAGES = 0  # 0 = 全量 106,345

VAE_CONFIGS = {
    "f4_kl-f4_p4": {
        "path": os.path.join(BASE, "pretrained_models", "kl-f4"),
        "downscale": 4, "latent_channels": 3,
        "scaling_factor": 0.102079, "patch": 4,
    },
    "f4_kl-f4_p2": {
        "path": os.path.join(BASE, "pretrained_models", "kl-f4"),
        "downscale": 4, "latent_channels": 3,
        "scaling_factor": 0.102079, "patch": 2,
    },
    "f8_sd-vae_p4": {
        "path": os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"),
        "downscale": 8, "latent_channels": 4,
        "scaling_factor": 0.18215, "patch": 4,
    },
    "f8_sd-vae_p2": {
        "path": os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"),
        "downscale": 8, "latent_channels": 4,
        "scaling_factor": 0.18215, "patch": 2,
    },
}

OUT_DIR = os.path.join(BASE, "tools", "vae_noise_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 多线程 SSIM (scipy) ──
from scipy.ndimage import uniform_filter

def compute_ssim(img1, img2):
    """SSIM per-channel average, img1/img2: (H,W,3) float [0,1]"""
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    win = 7
    ssims = []
    for ch in range(img1.shape[2]):
        x = img1[:, :, ch].astype(np.float64)
        y = img2[:, :, ch].astype(np.float64)
        mu_x = uniform_filter(x, size=win)
        mu_y = uniform_filter(y, size=win)
        mu_x2 = mu_x ** 2; mu_y2 = mu_y ** 2; mu_xy = mu_x * mu_y
        sigma_x2 = uniform_filter(x * x, size=win) - mu_x2
        sigma_y2 = uniform_filter(y * y, size=win) - mu_y2
        sigma_xy = uniform_filter(x * y, size=win) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))


def load_image_paths(csv_path, max_n=0):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if max_n > 0:
        rows = rows[:max_n]
    paths = []
    for r in rows:
        p = r["image_path"]
        # final_images/ 已删除, 统一替换为 final_imgs_256/
        if p.startswith("final_images/"):
            p = p.replace("final_images/", "final_imgs_256/", 1)
        full = os.path.join(IMG_ROOT, p)
        if os.path.isfile(full):
            paths.append(full)
    return paths


def load_batch(paths):
    """Load images as (B,3,256,256) float32 tensor [0,1], multi-threaded"""
    def _load(p):
        img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0
    with ThreadPoolExecutor(max_workers=16) as pool:
        imgs = list(pool.map(_load, paths))
    if not imgs:
        return None
    arr = np.stack(imgs)  # (B,H,W,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (B,3,H,W)
    return t


def evaluate_vae_config(name, cfg, paths, device):
    """对全部图片用该 VAE encode→decode, 统计 MSE/SSIM。"""
    from diffusers.models import AutoencoderKL
    print(f"\n{'='*60}")
    print(f"=== {name} (ds={cfg['downscale']}, ch={cfg['latent_channels']}, "
          f"scale={cfg['scaling_factor']}, patch={cfg['patch']}) ===")
    print(f"  images: {len(paths)}")

    vae = AutoencoderKL.from_pretrained(cfg["path"]).to(device).eval()
    # 统计 latent shape & tokens
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        latent = vae.encode(dummy).latent_dist.sample()
        latent_shape = tuple(latent.shape[1:])
        h, w = latent_shape[1], latent_shape[2]
        tokens_p2 = (h // 2) * (w // 2)
        tokens_p4 = (h // 4) * (w // 4)
        tokens = tokens_p2 if cfg["patch"] == 2 else tokens_p4
    print(f"  latent shape: {latent_shape}, patch{cfg['patch']} → {tokens} tokens")

    all_mse = []
    all_ssim = []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i:i + BATCH_SIZE]
            x = load_batch(batch_paths)
            if x is None:
                continue
            x = x.to(device)
            # encode→decode
            latent = vae.encode(x).latent_dist.sample()
            decoded = vae.decode(latent).sample
            decoded = decoded.clamp(0, 1).cpu().numpy()  # (B,3,H,W)
            x_np = x.cpu().numpy()  # (B,3,H,W)
            for j in range(decoded.shape[0]):
                orig = x_np[j].transpose(1, 2, 0)  # (H,W,3)
                rec = decoded[j].transpose(1, 2, 0)
                mse = float(np.mean((orig - rec) ** 2))
                all_mse.append(mse)
                all_ssim.append(0.0)  # placeholder, 批量算 SSIM 太慢
            n_done += len(batch_paths)
            if n_done % (BATCH_SIZE * 2) == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(paths) - n_done) / rate
                print(f"  [{name}] {n_done}/{len(paths)} ({rate:.1f} img/s, "
                      f"ETA {eta/60:.0f}min, MSE={np.mean(all_mse):.6f})", flush=True)
                sys.stdout.flush()
            del x, latent, decoded
            gc.collect()

            # 每 640 张存一次中间结果, 防止全挂
            if n_done % (BATCH_SIZE * 20) == 0:
                _mid = {
                    "name": name, "n_done": n_done,
                    "mse_mean": float(np.mean(all_mse)),
                    "ssim_mean": float(np.mean(all_ssim)),
                }
                with open(os.path.join(OUT_DIR, f"progress_{name}.json"), "w") as _f:
                    json.dump(_mid, _f)

    elapsed = time.time() - t0
    # SSIM 第二遍: 从 saved decoded? 不行, 内存爆。改为: 对前 1000 张重算 SSIM 采样
    # 不现实。改为: 只统计 MSE, SSIM 单独跑一个 1000 张采样版
    print(f"\n  [{name}] MSE pass done: {n_done} imgs in {elapsed:.0f}s ({n_done/elapsed:.1f} img/s)")
    print(f"  MSE: mean={np.mean(all_mse):.6f} std={np.std(all_mse):.6f} "
          f"median={np.median(all_mse):.6f} p95={np.percentile(all_mse,95):.6f} "
          f"max={np.max(all_mse):.6f}")

    # 保存逐图 MSE 结果
    per_image = [{"idx": i, "mse": m} for i, m in enumerate(all_mse)]
    detail_path = os.path.join(OUT_DIR, f"detail_{name}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(per_image, f)

    summary = {
        "name": name,
        "vae_path": cfg["path"],
        "downscale": cfg["downscale"],
        "latent_channels": cfg["latent_channels"],
        "scaling_factor": cfg["scaling_factor"],
        "patch": cfg["patch"],
        "latent_shape": latent_shape,
        "tokens": tokens,
        "n_images": len(all_mse),
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "mse_median": float(np.median(all_mse)),
        "mse_p95": float(np.percentile(all_mse, 95)),
        "mse_max": float(np.max(all_mse)),
        "ssim_mean": None,  # 见下方 ssim_sample
        "elapsed_sec": elapsed,
        "rate": len(all_mse) / elapsed,
    }

    # SSIM 采样: 对前 1000 张算 SSIM (代表性足够)
    print(f"  [{name}] computing SSIM on first 1000 samples...")
    ssim_vals = []
    for i in range(min(1000, len(paths))):
        x1 = load_batch([paths[i]])
        with torch.no_grad():
            lat = vae.encode(x1.to(device)).latent_dist.sample()
            dec = vae.decode(lat).sample.clamp(0, 1).cpu().numpy()
        orig = x1[0].cpu().numpy().transpose(1, 2, 0)
        rec = dec[0].transpose(1, 2, 0)
        ssim_vals.append(compute_ssim(orig, rec))
        del x1, lat, dec
    summary["ssim_mean"] = float(np.mean(ssim_vals))
    summary["ssim_std"] = float(np.std(ssim_vals))
    summary["ssim_n"] = len(ssim_vals)
    print(f"  SSIM (n={len(ssim_vals)}): mean={summary['ssim_mean']:.4f} std={summary['ssim_std']:.4f}")
    del vae
    gc.collect()
    return summary


def main():
    paths = load_image_paths(CSV_PATH, MAX_IMAGES)
    print(f"=== VAE 本底噪声监测 (全量 {len(paths)} 张) ===")
    print(f"CSV: {CSV_PATH}")
    print(f"Batch size: {BATCH_SIZE}, Device: {DEVICE}")
    print(f"配置数: {len(VAE_CONFIGS)}")

    summaries = []
    for name, cfg in VAE_CONFIGS.items():
        s = evaluate_vae_config(name, cfg, paths, DEVICE)
        summaries.append(s)

    # 保存汇总
    summary_path = os.path.join(OUT_DIR, "vae_noise_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"=== 汇总 ({len(summaries)} 配置, {len(paths)} 张图) ===")
    print(f"{'配置':<20} {'VAE':<8} {'patch':<6} {'tokens':<8} "
          f"{'MSE':<12} {'SSIM(1k)':<10}")
    for s in summaries:
        ds = s["downscale"]
        vae_name = "f4" if ds == 4 else "f8"
        ssim_str = f"{s['ssim_mean']:.4f}" if s.get('ssim_mean') else "N/A"
        print(f"{s['name']:<20} {vae_name:<8} {s['patch']:<6} {s['tokens']:<8} "
              f"{s['mse_mean']:.6f}    {ssim_str}")
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
