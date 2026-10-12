#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vae_three_way.py — SD-VAE / FLUX-16ch VAE / Calli-VAE 三方对照（实测）

回答两个问题：
  A. "KL 权重高还是低" —— 用【后验相对宽度】实测（scale-invariant）：
        w = mean_per_image_posterior_std / aggregate_latent_std
     w→0 = 潜码确定（mode≈sample）；w 大 = 潜码随机（mode≠sample）
  B. "FLUX 16ch 当年端到端为什么崩" —— 验证 shift_factor 是否被漏掉：
     分别用 decode(z/sf + shift) 与 decode(z/sf) 做往返，比 PSNR。

同时给出三方的 往返 PSNR/SSIM/墨量/高频（判断谁的细节保真最好）。
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
VAES = [
    ("sd-vae-ft-ema (4ch)",  f"{ROOT}/data/pretrained/pretrained_models/sd-vae-ft-ema", 0.18215, 0.0, "mode"),
    ("calli-vae kl1e6 (4ch)", f"{ROOT}/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv", 0.18215, 0.0, "sample"),
    ("flux-vae (16ch)",      f"{ROOT}/data/pretrained/flux_vae_eval", 0.3611, 0.1159, "sample"),
]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cuda:0 会与训练抢显存；测量很小，默认 CPU")
    ap.add_argument("--batch", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device(a.device)
    use_amp = dev.type == "cuda"
    gts = []
    for i in range(12):
        p = os.path.join(D, f"gt{i}.png")
        if os.path.exists(p):
            gts.append(np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)))
    g = np.stack(gts)
    gs = np.mean([stats(x) for x in g], axis=0)
    print(f"[data] {len(g)} 张 GT   ink={gs[0]:.4f} grad={gs[1]:.4f} hf={gs[3]:.4f}")
    xt = torch.from_numpy(g).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0

    from diffusers.models import AutoencoderKL
    print(f"\n{'VAE':<24}{'latent':>10}{'后验std(相对)':>14}{'聚合std':>9} | "
          f"{'PSNR':>7}{'SSIM':>7}{'ink×GT':>8}{'hf×GT':>7}")
    for name, path, sf, shift, mode in VAES:
        if not os.path.isdir(path):
            print(f"{name:<24}  (缺)"); continue
        vae = AutoencoderKL.from_pretrained(path).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        mus, sds, zs = [], [], []
        n = len(g)
        for i in range(0, n, a.batch):
            xb = xt[i:i + a.batch].to(dev)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()
            with torch.no_grad(), ctx:
                post = vae.encode(xb).latent_dist
                mus.append(post.mean.float().cpu())
                sds.append((post.std.float() if hasattr(post, "std") else torch.exp(0.5 * post.logvar).float()).cpu())
                zs.append((post.sample() if mode == "sample" else post.mode()).float().cpu())
        mu = torch.cat(mus); sd = torch.cat(sds); z_raw = torch.cat(zs)
        post_rel = float(sd.mean() / z_raw.std())
        agg = float(z_raw.std())
        lat = tuple(z_raw.shape[1:])
        # ★ 模拟真实链路: 落盘 z_stored = (z_raw - shift)*sf ; 解码 z_stored/sf + shift
        z_stored = (z_raw - shift) * sf
        recs = []
        for i in range(0, n, a.batch):
            zb = (z_stored[i:i + a.batch] / sf + shift).to(dev)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()
            with torch.no_grad(), ctx:
                recs.append(vae.decode(zb).sample.float().cpu())
        rec = torch.cat(recs)
        arr = np.clip(((rec.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).numpy(), 0, 255).astype(np.uint8)
        m = np.mean([stats(x) for x in arr], axis=0)
        P = np.mean([psnr(arr[i], g[i]) for i in range(len(g))])
        S = np.mean([ssimg(np.asarray(Image.fromarray(arr[i]).convert("L"), np.float32),
                           np.asarray(Image.fromarray(g[i]).convert("L"), np.float32)) for i in range(len(g))])
        print(f"{name:<24}{str(lat):>10}{post_rel:>14.5f}{agg:>9.3f} | "
              f"{P:>7.2f}{S:>7.4f}{m[0]/gs[0]:>8.2f}{m[3]/gs[3]:>7.2f}", flush=True)
        if shift:
            recs2 = []
            for i in range(0, n, a.batch):
                zb = (z_stored[i:i + a.batch] / sf).to(dev)      # ← 漏掉 +shift
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()
                with torch.no_grad(), ctx:
                    recs2.append(vae.decode(zb).sample.float().cpu())
            rec2 = torch.cat(recs2)
            arr2 = np.clip(((rec2.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).numpy(), 0, 255).astype(np.uint8)
            P2 = np.mean([psnr(arr2[i], g[i]) for i in range(len(g))])
            print(f"{'  └ 若漏掉 +shift':<24}{'':>10}{'':>14}{'':>9} | PSNR={P2:>6.2f}  (与上式对比)")
        del vae
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print("\n判读: 后验std(相对) 越小 = 潜码越确定(mode≈sample)。FLUX 的 +shift 若漏掉, PSNR 会崩 -> 即 v63b 失败根因。")


if __name__ == "__main__":
    main()
