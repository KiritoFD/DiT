#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 f8(sd-vae-ft-ema)+patch2 vs f4(kl-f4)+patch4 的 VAE 重建质量。
两个配置都给 DiT 256 个 token，每个 token 覆盖 16x16 原图像素：
  - f8: 256→32 latent, patch2 → 16x16=256 tokens, 每 token = 2x2x3=12 latent 值
  - f4: 256→64 latent, patch4 → 16x16=256 tokens, 每 token = 4x4x3=48 latent 值
关键: VAE 是最终画质的瓶颈 — DiT 再好也重建不出 VAE 丢掉的高频细节。
本脚本: 取 N 张书法图 → 两个 VAE 各自 encode+decode → 算 MSE/SSIM/LPIPS。
"""
import os, sys, csv, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ── 配置 ──
BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(BASE, "5script", "eval100_top30_clean.csv")  # 79 张干净评测图
IMG_ROOT = BASE  # CSV 里是 final_imgs_256/xxx.png
N = 100  # 用前 N 张
DEVICE = "cpu"  # 远程 CPU，不干扰 GPU 训练

VAE_CONFIGS = {
    "f8_sd-vae-ft-ema": {
        "path": os.path.join(BASE, "pretrained_models", "sd-vae-ft-ema"),
        "downscale": 8,
        "latent_channels": 4,
        "scaling_factor": 0.18215,
    },
    "f4_kl-f4": {
        "path": os.path.join(BASE, "pretrained_models", "kl-f4"),
        "downscale": 4,
        "latent_channels": 3,
        "scaling_factor": 0.102079,
    },
}


def load_vae(path, device):
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(path).to(device).eval()
    return vae


def load_images(n):
    """加载前 n 张图，返回 list of (img_id, PIL.Image 256x256 RGB)"""
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    images = []
    for r in rows[:n]:
        p = os.path.join(IMG_ROOT, r["image_path"])
        if os.path.isfile(p):
            img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
            images.append((r["image_path"], img))
    return images


def to_tensor(img, device):
    """PIL → (1,3,256,256) normalized to [0,1]"""
    a = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def compute_ssim(img1, img2):
    """SSIM per-channel average, img1/img2: (H,W,3) float [0,1]. 纯 numpy 实现。"""
    c1, c2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    from scipy.ndimage import uniform_filter
    win = 7
    ssims = []
    for ch in range(img1.shape[2]):
        x = img1[:, :, ch].astype(np.float64)
        y = img2[:, :, ch].astype(np.float64)
        mu_x = uniform_filter(x, size=win)
        mu_y = uniform_filter(y, size=win)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sigma_x2 = uniform_filter(x * x, size=win) - mu_x2
        sigma_y2 = uniform_filter(y * y, size=win) - mu_y2
        sigma_xy = uniform_filter(x * y, size=win) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))


def compute_mse(img1, img2):
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))


def evaluate_vae(vae, cfg, images, device):
    """对每张图: encode→decode→比较重建质量。同时记录 latent shape。"""
    results = []
    t0 = time.time()
    with torch.no_grad():
        for i, (img_id, pil_img) in enumerate(images):
            x = to_tensor(pil_img, device)
            # encode
            latent = vae.encode(x).latent_dist.sample()
            latent_shape = tuple(latent.shape[1:])  # (C, H, W)
            # decode
            decoded = vae.decode(latent).sample
            decoded = decoded.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
            orig = np.asarray(pil_img, dtype=np.float32) / 255.0
            mse = compute_mse(orig, decoded)
            ssim_val = compute_ssim(orig, decoded)
            results.append({
                "img_id": img_id,
                "mse": mse,
                "ssim": ssim_val,
                "latent_shape": latent_shape,
            })
            if (i + 1) % 20 == 0:
                print(f"  [{cfg['name']}] {i+1}/{len(images)} ...", flush=True)
    elapsed = time.time() - t0
    return results, elapsed


def main():
    print(f"=== VAE 重建质量对比 (f8+patch2 vs f4+patch4, 同 256 token) ===")
    print(f"图片: {CSV_PATH}, 前 {N} 张")
    print(f"设备: {DEVICE}")
    print()

    images = load_images(N)
    print(f"加载 {len(images)} 张图")

    all_results = {}
    for name, cfg in VAE_CONFIGS.items():
        cfg["name"] = name
        print(f"\n--- {name} (downscale={cfg['downscale']}, latent_ch={cfg['latent_channels']}) ---")
        vae = load_vae(cfg["path"], DEVICE)
        results, elapsed = evaluate_vae(vae, cfg, images, DEVICE)
        mses = [r["mse"] for r in results]
        ssims = [r["ssim"] for r in results]
        latent_shape = results[0]["latent_shape"] if results else None
        # token 数
        if latent_shape:
            h, w = latent_shape[1], latent_shape[2]
            tokens_patch2 = (h // 2) * (w // 2)
            tokens_patch4 = (h // 4) * (w // 4)
        else:
            tokens_patch2 = tokens_patch4 = 0
        all_results[name] = {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
            "latent_shape": latent_shape,
            "tokens_patch2": tokens_patch2,
            "tokens_patch4": tokens_patch4,
            "elapsed": elapsed,
            "results": results,
        }
        print(f"  MSE:  {np.mean(mses):.6f} ± {np.std(mses):.6f}")
        print(f"  SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        print(f"  latent shape: {latent_shape}")
        print(f"  tokens (patch2): {tokens_patch2}, tokens (patch4): {tokens_patch4}")
        print(f"  time: {elapsed:.1f}s ({len(images)/elapsed:.1f} img/s)")
        del vae

    # 总结
    print("\n" + "=" * 60)
    print("=== 总结 ===")
    print(f"{'配置':<25} {'MSE':<12} {'SSIM':<12} {'latent':<15} {'patch→tokens':<20}")
    for name, r in all_results.items():
        ds = VAE_CONFIGS[name]["downscale"]
        patch = 2 if ds == 8 else 4
        toks = r["tokens_patch2"] if ds == 8 else r["tokens_patch4"]
        print(f"{name:<25} {r['mse_mean']:.6f}    {r['ssim_mean']:.4f}      {str(r['latent_shape']):<15} p{patch}→{toks}")

    # 判定
    names = list(all_results.keys())
    r0, r1 = all_results[names[0]], all_results[names[1]]
    if r0["ssim_mean"] > r1["ssim_mean"]:
        better, worse = names[0], names[1]
    else:
        better, worse = names[1], names[0]
    print(f"\n重建质量更好: {better}")
    print(f"  {better}: SSIM={all_results[better]['ssim_mean']:.4f}, MSE={all_results[better]['mse_mean']:.6f}")
    print(f"  {worse}: SSIM={all_results[worse]['ssim_mean']:.4f}, MSE={all_results[worse]['mse_mean']:.6f}")
    diff_ssim = abs(r0["ssim_mean"] - r1["ssim_mean"])
    print(f"  SSIM 差距: {diff_ssim:.4f}")

    # 保存
    out_path = os.path.join(BASE, "tools", "vae_recon_compare.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "results"} 
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
