#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf_decompose.py — 同一批字上的 HF 分解（回答"高频到底是 VAE 还是 DiT 的问题"）

之前的漏洞: VAE 天花板 0.78 是在 12 字子集上测的, 而 DiT 的 0.51 是在另一个 40 字子集上测的
            -> 两个数不同源, 分解不成立。本脚本在**同一批 187 字**上同时测:

  HF(GT)                        —— 基准
  HF(decode(encode(GT)))        —— **VAE 往返天花板**（每个行用它自己那个 VAE）
  HF(DiT 输出)                  —— eval/<row>/{i}.png

输出关键量: **DiT 用掉了自己 VAE 天花板的百分之几** = HF_dit / HF_roundtrip
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SD_VAE = "pretrained_models/sd-vae-ft-ema"
CALLI_VAE = "experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"
TOPBAND_VAE = "experiments/vae_fix_topband/calli_vae_final"
# ★ 2026-10-11: 纯物理 VAE（去 DINO: L1+LPIPS+PatchGAN+KL+topband+Laplacian, 20k 步）
#   验收: 往返顶带 0.901 / LPIPS 0.0135 / PSNR 34.04（均优于 SD 与旧 DINO 版）
PHYSICS_VAE = "experiments/calli_vae_physics/calli_vae_final"

# 行 -> 该行**实际使用的** VAE（决定它的往返天花板）
ROWS = {
    "v68": SD_VAE, "v68_sde": SD_VAE,
    "v72": CALLI_VAE, "v72_sde": CALLI_VAE,
    "v76": TOPBAND_VAE, "v77": TOPBAND_VAE, "v78": TOPBAND_VAE,
    "v79": PHYSICS_VAE, "v79_sde": PHYSICS_VAE,
}
# 需要测往返的 VAE 集合 (tag, path, mode)
CEIL_VAES = [
    ("vae_sd", SD_VAE, "mode"),
    ("vae_calli", CALLI_VAE, "sample"),
    ("vae_topband", TOPBAND_VAE, "sample"),
    ("vae_physics", PHYSICS_VAE, "sample"),
]
TAG_OF = {SD_VAE: "vae_sd", CALLI_VAE: "vae_calli", TOPBAND_VAE: "vae_topband",
          PHYSICS_VAE: "vae_physics"}


def hf_of(arr_uint8, r_cut=0.5):
    """高频能量占比（径向 r>r_cut 的 FFT 能量份额）—— 与 census / 训练内 band_fraction 同口径。"""
    a = arr_uint8.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    b = a / 255.0
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return float(F[r > r_cut].sum() / max(F.sum(), 1e-9))


def ink_of(arr_uint8):
    a = arr_uint8.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    return float((a < 128).mean())


def load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=187)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device(a.device)

    gt = [load(f"eval/gt/{i}.png") for i in range(a.n)]
    hf_gt = np.array([hf_of(x) for x in gt])
    ink_gt = np.array([ink_of(x) for x in gt])
    print(f"[GT] n={len(gt)}  hf={hf_gt.mean():.4f}  ink={ink_gt.mean():.4f}")

    from diffusers.models import AutoencoderKL
    rt = {}
    for tag, vpath, mode in CEIL_VAES:
        if not os.path.isdir(vpath):
            print(f"[{tag}] 缺 {vpath}"); continue
        vae = AutoencoderKL.from_pretrained(vpath).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        outs = []
        for i in range(0, a.n, a.batch):
            xb = torch.from_numpy(np.stack(gt[i:i + a.batch])).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.no_grad()
            with torch.no_grad(), amp:
                post = vae.encode(xb).latent_dist
                z = post.mode() if mode == "mode" else post.sample()
                rec = vae.decode(z).sample.float()
            arr = np.clip(((rec.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy(), 0, 255).astype(np.uint8)
            outs.extend(list(arr))
        rt[tag] = (np.array([hf_of(x) for x in outs]), np.array([ink_of(x) for x in outs]))
        print(f"[{tag}] 往返 hf={rt[tag][0].mean():.4f} (={rt[tag][0].mean()/hf_gt.mean():.3f}x GT)  "
              f"ink={rt[tag][1].mean():.4f} (={rt[tag][1].mean()/ink_gt.mean():.3f}x GT)")
        del vae
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print(f"{'行':<10}{'hf':>8}{'hf/GT':>7}{'ink':>8}{'ink/GT':>8}   | 用掉 VAE 天花板 (hf_dit/hf_rt)")
    for row, vpath in ROWS.items():
        d = f"eval/{row}"
        if not os.path.isdir(d):
            print(f"{row:<10}  (缺图)"); continue
        imgs = [load(f"{d}/{i}.png") for i in range(a.n) if os.path.exists(f"{d}/{i}.png")]
        if not imgs:
            print(f"{row:<10}  (空)"); continue
        hf = np.array([hf_of(x) for x in imgs])
        ink = np.array([ink_of(x) for x in imgs])
        tag = TAG_OF.get(vpath)
        ceil = rt[tag][0].mean() if tag in rt else float("nan")
        print(f"{row:<10}{hf.mean():>8.4f}{hf.mean()/hf_gt.mean():>7.3f}{ink.mean():>8.4f}"
              f"{ink.mean()/ink_gt.mean():>8.3f}   | {hf.mean()/ceil:>6.3f}  (天花板 {ceil:.4f})")

    print()
    print(f"逐图配对 (n={a.n}):  HF_dit / HF_roundtrip  中位 / p10 / p90")
    for row, vpath in ROWS.items():
        d = f"eval/{row}"
        if not os.path.isdir(d):
            continue
        hf = np.array([hf_of(load(f"{d}/{i}.png")) for i in range(a.n) if os.path.exists(f"{d}/{i}.png")])
        tag = TAG_OF.get(vpath)
        if tag not in rt:
            continue
        ceil = rt[tag][0][:len(hf)]
        ratio = hf / np.maximum(ceil, 1e-9)
        print(f"  {row:<10} 中位 {np.median(ratio):.3f}   p10 {np.percentile(ratio,10):.3f}   p90 {np.percentile(ratio,90):.3f}")


if __name__ == "__main__":
    main()
