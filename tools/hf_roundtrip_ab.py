#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf_roundtrip_ab.py — 独立复测: 冻结 VAE vs 臂B(HF修正) VAE 的往返高频/墨量/L1
同一批 187 张 eval/gt/*.png。用于验证"HFr 0.78 -> ~1.0"这个关键主张(不只信训练内自报)。
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAES = [
    ("frozen  (v72用)", "experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"),
    ("armB_hffix       ", "experiments/vae_fix_hf/calli_vae_final"),
]
N, BATCH = 187, 4


def hf_of(a):
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    b = a / 255.0
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return float(F[r > 0.5].sum() / max(F.sum(), 1e-9))


def ink_of(a):
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    return float((a < 128).mean())


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gt = [np.asarray(Image.open(f"eval/gt/{i}.png").convert("RGB").resize((256, 256), Image.BICUBIC))
          for i in range(N)]
    hf_gt = np.mean([hf_of(x) for x in gt])
    ink_gt = np.mean([ink_of(x) for x in gt])
    print(f"[GT] n={N}  hf={hf_gt:.4f}  ink={ink_gt:.4f}")

    from diffusers.models import AutoencoderKL
    print(f"\n{'VAE':<20}{'往返hf':>9}{'hf/GT':>8}{'往返ink':>9}{'ink/GT':>8}{'L1':>8}{'posterior_std':>15}")
    for name, path in VAES:
        if not os.path.isdir(path):
            print(f"{name:<20}  (缺 {path})"); continue
        vae = AutoencoderKL.from_pretrained(path).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        outs, l1s, sds = [], [], []
        for i in range(0, N, BATCH):
            xb = torch.from_numpy(np.stack(gt[i:i + BATCH])).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.no_grad():
                post = vae.encode(xb).latent_dist
                sds.append(float((post.std if hasattr(post, "std") else torch.exp(0.5 * post.logvar)).float().mean()))
                z = post.sample()
                rec = vae.decode(z).sample.float()
            arr = np.clip(((rec.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy(), 0, 255).astype(np.uint8)
            outs.extend(list(arr))
            l1s.append(float(np.abs(arr.astype(np.float32) / 255.0 - np.stack(gt[i:i + BATCH]).astype(np.float32) / 255.0).mean()))
        hf = np.mean([hf_of(x) for x in outs]); ink = np.mean([ink_of(x) for x in outs])
        print(f"{name:<20}{hf:>9.4f}{hf/hf_gt:>8.3f}{ink:>9.4f}{ink/ink_gt:>8.3f}{np.mean(l1s):>8.4f}{np.mean(sds):>15.4f}")
        del vae
        torch.cuda.empty_cache()
    print("\n判读: 若 armB 的 hf/GT 从 0.783 升到 ~1.0 -> 高频修正成功(独立复测通过)")


if __name__ == "__main__":
    main()
