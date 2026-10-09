#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""judge_vae.py — VAE 判定器：后验宽度 + 往返保真（墨量/高频/PSNR/SSIM），一切以 GT 为基准。

用法（可给多个候选一起评）:
  python tools/vae/judge_vae.py \
    --spec "calli(sample)=experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv:0.18215:0:mode" \
    --spec "fixC(finetune)=experiments/vae_fix_both/calli_vae_step_01500:0.18215:0:sample" \
    --spec "flux16=...:0.3611:0.1159:sample"

--spec 格式: 名称=路径:scaling:shift:mode(mode|sample)
判据（三个数）:
  ① 后验相对宽度 → 应趋近 0（SD 4e-5 / FLUX 2e-4 / 我们 0.372）
  ② 往返 高频×GT  → 应从 0.78 升向 ≥0.85（4ch 上限参考 SD 0.81）
  ③ PSNR 不应明显下降（≤0.3 dB）
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
D = f"{ROOT}/experiments/poster_4col/v68_g0.0/eval_samples_ctrl/step0200000/poster"


def stats(a):
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    ink = float((a < 128).mean())
    b = a / 255.0
    gx = np.zeros_like(b); gy = np.zeros_like(b)
    gx[:, 1:-1] = b[:, 2:] - b[:, :-2]
    gy[1:-1, :] = b[2:, :] - b[:-2, :]
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    lap = np.zeros_like(b)
    lap[1:-1, 1:-1] = (4 * b[1:-1, 1:-1] - b[:-2, 1:-1] - b[2:, 1:-1] - b[1:-1, :-2] - b[1:-1, 2:])
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return ink, grad, float(lap.var()), float(F[r > 0.5].sum() / max(F.sum(), 1e-9))


def psnr(a, b):
    m = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if m <= 1e-12 else float(10 * np.log10(255.0 * 255.0 / m))


def ssimg(a, b):
    from skimage.metrics import structural_similarity as f
    return float(f(a.astype(np.float32), b.astype(np.float32), data_range=255.0))


def parse_spec(s):
    name, _, rest = s.partition("=")
    parts = rest.split(":")
    path, sf, shift, mode = parts[0], float(parts[1]), float(parts[2]), (parts[3] if len(parts) > 3 else "sample")
    return name, path, sf, shift, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    dev = torch.device(a.device)
    use_amp = dev.type == "cuda"

    gts = []
    for i in range(a.n):
        p = os.path.join(D, f"gt{i}.png")
        if os.path.exists(p):
            gts.append(np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)))
    g = np.stack(gts)
    gs = np.mean([stats(x) for x in g], axis=0)
    print(f"[GT] n={len(g)}  ink={gs[0]:.4f}  grad={gs[1]:.4f}  hf={gs[3]:.4f}")
    xt = torch.from_numpy(g).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0

    from diffusers.models import AutoencoderKL
    print(f"\n{'VAE':<28}{'后验std(相对)':>14}{'PSNR':>8}{'SSIM':>8}{'ink×GT':>8}{'hf×GT':>7}  判定")
    for s in a.spec:
        name, path, sf, shift, mode = parse_spec(s)
        if not os.path.isdir(path):
            print(f"{name:<28}  (缺) {path}"); continue
        vae = AutoencoderKL.from_pretrained(path).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()):
            post = vae.encode(xt).latent_dist
            sd = post.std.float() if hasattr(post, "std") else torch.exp(0.5 * post.logvar).float()
            z = (post.sample() if mode == "sample" else post.mode()).float()
        post_rel = float(sd.mean() / z.std())
        z_stored = (z - shift) * sf
        with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()):
            rec = vae.decode((z_stored / sf + shift).to(dev)).sample.float().cpu()
        arr = np.clip(((rec.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).numpy(), 0, 255).astype(np.uint8)
        m = np.mean([stats(x) for x in arr], axis=0)
        P = np.mean([psnr(arr[i], g[i]) for i in range(len(g))])
        S = np.mean([ssimg(np.asarray(Image.fromarray(arr[i]).convert("L"), np.float32),
                           np.asarray(Image.fromarray(g[i]).convert("L"), np.float32)) for i in range(len(g))])
        hfr = m[3] / gs[3]
        ok_c = "✓" if post_rel < 0.05 else "✗"
        ok_h = "✓" if hfr >= 0.85 else ("~" if hfr > 0.78 else "✗")
        print(f"{name:<28}{post_rel:>14.5f}{P:>8.2f}{S:>8.4f}{m[0]/gs[0]:>8.2f}{hfr:>7.2f}  "
              f"坍缩{ok_c} 高频{ok_h}")
        del vae
    print("\n参考: SD 后验4e-5/高频0.81 | FLUX-16ch 2e-4/0.97 | 我们(冻结版) 0.372/0.78")


if __name__ == "__main__":
    main()
