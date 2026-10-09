#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vae_roundtrip_ink.py — 判断"欠墨"是 VAE 的锅还是 DiT 的锅。

做法：拿评测集里的真迹图，直接做 VAE 往返（encode → decode），量重建图的墨量/PSNR/SSIM。
  * 若往返就掉墨（GT≈0.226 -> 重建≈0.15）  => VAE/解码约定在丢墨, 两个模型都瘦同源
  * 若往返保墨（≈0.22）                      => DiT 学出来的欠墨, 属训练目标/损失问题

同时给出"约定对照"：正确约定 decode(z) vs 旧错误约定 decode(z/sf)。
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
SD_VAE = f"{ROOT}/data/pretrained/pretrained_models/sd-vae-ft-ema"
CALLI_VAE = f"{ROOT}/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"
CSV = f"{ROOT}/exp-std/csv/poster12.csv"


def ink(a_uint8):
    """暗像素占比（<128）与对比度。"""
    g = a_uint8.astype(np.float32)
    return float((g < 128).mean()), float(g.mean()), float(g.std())


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 * 255.0 / mse))


def ssim_gray(a, b):
    from skimage.metrics import structural_similarity as f
    return float(f(a.astype(np.float32), b.astype(np.float32), data_range=255.0))


def load_gts(n):
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    paths = []
    for r in rows[:n]:
        p = r.get("image_path") or r.get("img_path") or r.get("path") or ""
        if not os.path.isabs(p):
            p = os.path.join(ROOT, p)
        if os.path.exists(p):
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    paths = load_gts(a.n)
    print(f"[data] {len(paths)} 张真迹: {[os.path.basename(p) for p in paths[:3]]} ...")
    if not paths:
        print("  没有可用图，退出"); return

    gts = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)
        gts.append(np.asarray(im))
    g = np.stack(gts)
    g_ink, g_mean, g_std = np.mean([ink(x) for x in g], axis=0)
    print(f"\n[GT]        墨量={g_ink:.4f}  均值={g_mean:.1f}  std={g_std:.1f}   (n={len(g)})")

    from diffusers.models import AutoencoderKL

    for name, vpath in (("sd-vae-ft-ema", SD_VAE), ("calli-vae(kl1e6)", CALLI_VAE)):
        if not os.path.isdir(vpath):
            print(f"\n[{name}] 路径不存在: {vpath}"); continue
        vae = AutoencoderKL.from_pretrained(vpath).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        xt = torch.from_numpy(g).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            post = vae.encode(xt).latent_dist
            z_mode = post.mode()
            z_samp = post.sample()
        variants = {
            "decode(mode)   [SD 训练/推理约定]": z_mode,
            "decode(sample) [Calli 训练/推理约定]": z_samp,
            "decode(mode/sf)   [旧 calli 错误约定]": z_mode / 0.18215,
        }
        print(f"\n[{name}]")
        for tag, z in variants.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rec = vae.decode(z).sample.float()
            arr = ((rec.clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy()
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            inks = np.mean([ink(x) for x in arr], axis=0)
            ps = np.mean([psnr(arr[i], g[i]) for i in range(len(g))])
            ss = np.mean([ssim_gray(np.asarray(Image.fromarray(arr[i]).convert("L"),
                                             dtype=np.float32),
                                    np.asarray(Image.fromarray(g[i]).convert("L"), dtype=np.float32))
                          for i in range(len(g))])
            print(f"   {tag:<38} 墨量={inks[0]:.4f} ({inks[0]/g_ink:5.2f}x GT)  "
                  f"均值={inks[1]:6.1f} std={inks[2]:5.1f} | PSNR={ps:5.2f} SSIM={ss:.4f}")
        del vae
        torch.cuda.empty_cache()

    print("\n判读: 若两种约定的正确版本 墨量都 ≈ GT, 则欠墨来自 DiT; 若重建就掉墨, 则是 VAE 侧。")


if __name__ == "__main__":
    main()
