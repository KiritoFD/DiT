#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vae_roundtrip_sharp.py — VAE 往返是否丢高频？（决定"软"的责任在 VAE 还是 DiT）

对 12 张真迹做 encode->decode，量重建图的 高频占比/梯度/Laplacian 方差，与 GT 比。
"""
import os
import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
SD_VAE = f"{ROOT}/data/pretrained/pretrained_models/sd-vae-ft-ema"
CALLI_VAE = f"{ROOT}/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"
D = f"{ROOT}/experiments/poster_4col/v68_g0.0/eval_samples_ctrl/step0200000/poster"


def stats(a_uint8):
    a = a_uint8.astype(np.float32)
    if a.ndim == 3:                      # RGB -> 灰度（高频普查统一口径）
        a = a.mean(axis=-1)
    ink = float((a < 128).mean())
    b = a / 255.0
    gx = np.zeros_like(b); gy = np.zeros_like(b)
    gx[:, 1:-1] = b[:, 2:] - b[:, :-2]
    gy[1:-1, :] = b[2:, :] - b[:-2, :]
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    lap = np.zeros_like(b)
    lap[1:-1, 1:-1] = (4 * b[1:-1, 1:-1] - b[:-2, 1:-1] - b[2:, 1:-1] - b[1:-1, :-2] - b[1:-1, 2:])
    lapvar = float(lap.var())
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    hf = float(F[r > 0.5].sum() / max(F.sum(), 1e-9))
    return ink, grad, lapvar, hf


def main():
    dev = torch.device("cuda:0")
    gts = []
    for i in range(12):
        p = os.path.join(D, f"gt{i}.png")
        if os.path.exists(p):
            gts.append(np.asarray(Image.open(p).convert("RGB")))
    print(f"[data] {len(gts)} 张 GT")
    g = np.stack(gts)
    gs = np.mean([stats(x) for x in g], axis=0)
    print(f"\n{'':<18}{'ink':>8}{'grad':>8}{'lapvar':>10}{'hf':>8}")
    print(f"{'GT':<18}{gs[0]:>8.4f}{gs[1]:>8.4f}{gs[2]:>10.6f}{gs[3]:>8.4f}")

    from diffusers.models import AutoencoderKL
    xt = torch.from_numpy(g).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0
    for name, vpath in (("sd-vae(mode)", SD_VAE), ("calli(sample)", CALLI_VAE)):
        vae = AutoencoderKL.from_pretrained(vpath).to(dev).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            post = vae.encode(xt).latent_dist
            z = post.mode() if "mode" in name else post.sample()
            rec = vae.decode(z).sample.float()
        arr = ((rec.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        m = np.mean([stats(x) for x in arr], axis=0)
        print(f"{name:<18}{m[0]:>8.4f}{m[1]:>8.4f}{m[2]:>10.6f}{m[3]:>8.4f}   "
              f"(hf={m[3]/gs[3]:.2f}x GT, grad={m[1]/gs[1]:.2f}x)")
        del vae
        torch.cuda.empty_cache()

    print("\n判读: 若往返的 hf 也 ≈0.5x GT -> 软是 VAE 解码的锅; 若 ≈1.0x -> 是 DiT 学出来的。")


if __name__ == "__main__":
    main()
