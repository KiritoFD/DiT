#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_band_metric.py — 验证训练用的 band_fraction 与离线普查【同一口径】

判据: 用冻结 VAE 做往返, band_fraction(rec)/band_fraction(gt) 应当 ≈ 0.783
      （与 hf_decompose.py 的离线结果一致）。若一致 -> 训练日志的 HFr 从此可信。
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "vae"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hf_losses import band_fraction, hf_loss  # noqa: E402


def np_band_fraction(a_uint8, r_cut=0.5):
    """离线普查口径的 numpy 参考实现（应与 band_fraction 数值一致）。"""
    a = a_uint8.astype(np.float64)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    b = a / 255.0
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return float(F[r > r_cut].sum() / max(F.sum(), 1e-9))


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = 12
    gts = [np.asarray(Image.open(f"eval/gt/{i}.png").convert("RGB").resize((256, 256), Image.BICUBIC)) for i in range(N)]

    # 1) torch 版 vs numpy 版在 GT 上的数值一致性
    xt = torch.from_numpy(np.stack(gts)).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0
    bf_t = band_fraction(xt).mean().item()
    bf_n = np.mean([np_band_fraction(x) for x in gts])
    print(f"[口径一致性] GT:  torch band_fraction={bf_t:.4f}   numpy 普查={bf_n:.4f}   "
          f"相对差={abs(bf_t-bf_n)/bf_n*100:.2f}%")

    # 2) 冻结 VAE 往返 -> 比值应 ≈0.783
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv").to(dev).eval()
    outs = []
    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.no_grad()):
        post = vae.encode(xt).latent_dist
        z = post.sample()
        rec = vae.decode(z).sample.float()
    arr = np.clip(((rec.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy(), 0, 255).astype(np.uint8)
    rec_t = torch.from_numpy(arr).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0
    bf_rec_t = band_fraction(rec_t).mean().item()
    bf_rec_n = np.mean([np_band_fraction(x) for x in arr])
    print(f"[往返] GT={bf_t:.4f}  重建(torch)={bf_rec_t:.4f}  重建(numpy)={bf_rec_n:.4f}")
    print(f"[关键] band_fraction(rec)/band_fraction(gt) = {bf_rec_t/bf_t:.3f}   "
          f"（离线 hf_decompose 给出 0.783 -> 应一致）")

    # 3) 各模式损失的量级（便于设 w_hf）
    for m in ("topband", "dog", "fft", "focal"):
        v = hf_loss(rec_t, xt, m) if m != "topband" else hf_loss(rec_t, xt, m, r_cut=0.5)
        print(f"  hf_loss({m:8s}) = {float(v):.6f}")


if __name__ == "__main__":
    main()
