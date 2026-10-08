#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_vae_convention.py — 用实测判定一个 AutoencoderKL 的**解码约定**，并给出
四种解码输入各自的重建质量 (L1 / PSNR)。

为什么需要它 (2026-10-09)
------------------------
"latent 到底该喂 decode(z) 还是 decode(z/sf)?" 这件事**不能靠约定俗成去猜**。

SD-VAE 的既定约定是：后验采样得到原始潜变量 z，decoder 直接吃 z（`decode(z)`）；
diffusion 侧才把 z 乘上 scaling_factor 当训练目标（`z*sf`），推理时再除回来
（`decode(pred/sf) == decode(z)`）。

Calli-VAE 当年训练脚本写成 `decode(z / sf)`，等于把 decoder 的输入尺度放大了
1/0.18215 ≈ 5.49 倍，训出一个"只认 sample/sf"的非标准解码器。下游一旦按标准
约定 `decode(pred/sf)=decode(mode)` 去解，就得到灰图 (SSIM≈0.51, MSE≈0.79)。

本脚本对同一批图枚举 4 种解码输入，**L1 最低者就是该 VAE 的真实约定**：
    decode(sample)   decode(sample/sf)   decode(mode)   decode(mode/sf)

判定标准 (对标准 VAE)：
    decode(sample) 应该 ≈ decode(sample/sf) 且都很好（sf 只影响 mm 级）
    实际上正确组合通常是 decode(sample) 或 decode(mode)，视 VAE 是否被训歪。

用法
----
    python tools/vae/check_vae_convention.py --vae <vae_dir> \
        --csv <meta.csv> --data-root <root> [--n 8] [--size 256] [--sf 0.18215]

    # 无数据时用随机图做 sanity（不能判约定好坏，只验证能跑通）
    python tools/vae/check_vae_convention.py --vae <vae_dir> --random
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL


def load_images(args, device):
    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if args.random:
        return torch.rand(args.n, 3, args.size, args.size, device=device) * 2 - 1
    paths = []
    if args.img_dir:
        exts = (".png", ".jpg", ".jpeg", ".webp")
        for r, _d, fs in os.walk(args.img_dir):
            for fn in sorted(fs):
                if fn.lower().endswith(exts) and fn.startswith(args.prefix):
                    paths.append(os.path.join(r, fn))
            if len(paths) >= args.n:
                break
    else:
        with open(args.csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel = row.get("image_path") or row.get("path") or f"imgs/{row.get('img_id')}.png"
                paths.append(os.path.join(args.data_root, rel))
                if len(paths) >= args.n:
                    break
    paths = paths[:args.n]
    xs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append(tf(img))
    return torch.stack(xs).to(device)


def psnr(a, b):
    mse = F.mse_loss(a.clamp(-1, 1), b.clamp(-1, 1)).item()
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(4.0 / mse)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--img-dir", default="", help="直接扫描该目录下的真实图片 (优先于 --csv)")
    ap.add_argument("--prefix", default="", help="配合 --img-dir：只取该前缀的文件 (如 gt)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sf", type=float, default=0.18215)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[conv] VAE = {args.vae}")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    sf = float(vae.config.scaling_factor) if getattr(vae.config, "scaling_factor", None) else args.sf
    print(f"[conv] vae.config.scaling_factor = {sf}  (脚本传入 sf={args.sf})")

    x = load_images(args, device)
    post = vae.encode(x).latent_dist
    sample = post.sample()
    mode = post.mode() if hasattr(post, "mode") else post.mean
    print(f"[conv] posterior: mean={post.mean.mean().item():.4f} std={post.std.mean().item():.4f} "
          f"|sample-mose|={ (sample - mode).abs().mean().item():.4f}")

    combos = {
        "sample      (标准 z)": sample,
        "sample/sf            ": sample / sf,
        "mode        (确定性 mean)": mode,
        "mode/sf              ": mode / sf,
    }
    print("\n[conv] 解码输入 -> 重建质量 (L1/PSNR，越低/越高越好):")
    print("       {:<24} {:>10} {:>10}".format("decode(input)", "L1", "PSNR(dB)"))
    rows = []
    for name, z in combos.items():
        rec = vae.decode(z).sample
        l1 = F.l1_loss(rec, x).item()
        p = psnr(rec, x)
        rows.append((name, l1, p))
        print("       {:<24} {:>10.4f} {:>10.2f}".format(name, l1, p))

    best = min(rows, key=lambda r: r[1])
    print(f"\n[conv] 最优约定 = decode({best[0].strip().split()[0]})  (L1={best[1]:.4f})")
    # 标准 VAE 的判据：sample 与 sample/sf 都差 = 说明该 VAE 被训歪 / 约定非标准
    s_l1 = dict((r[0].strip().split()[0], r[1]) for r in rows)
    if s_l1.get("sample", 9) > 0.1 and s_l1.get("sample/sf", 9) < 0.05:
        print("[conv] ⚠ 该 VAE 只认 sample/sf —— 非标准约定 (典型的 decode(z/sf) 训练残留)。")
    elif s_l1.get("sample", 9) < 0.05:
        print("[conv] ✓ 该 VAE 符合标准约定 decode(sample)。")


if __name__ == "__main__":
    main()
