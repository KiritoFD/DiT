#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/bench_vae_trunc.py — VAE 解码截断到低分辨率: 显存省多少? 保真度够不够?

背景: 带梯度全分辨率(256x256)解码 = **1.45 GB/样本**, K 上限只有 8（不可用）。
峰值来自 decoder up_blocks[2]/[3] 在 256x256 的激活。

本脚本测两件事:
  A. 显存: decode 只跑到 up_blocks[k] (64/128/256 输出) + 一个 1x1 头 -> 3ch, 的边际显存
  B. 保真: 截断输出里还看不看得清"骨架"? (与真骨架比 IoU)
     —— 若 128x128 截断的骨架 IoU 够高, 它就能当损失空间; 否则不行。

⚠ 不开 expandable_segments。
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda'
GIB = 2 ** 30
TOTAL = 23.52


def gib(x):
    return x / GIB


def main():
    from src.model.deform_skel import DeformSkel
    from diffusers.models import AutoencoderKL
    from scipy.ndimage import label, generate_binary_structure
    from skimage.morphology import skeletonize

    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    for p in vae.parameters():
        p.requires_grad_(False)
    sc = float(getattr(vae.config, 'scaling_factor', 0.18215))
    dec = vae.decoder
    # 动态推断每个 stop 的输出通道数(别硬编码, SD-VAE 的 up_blocks 通道不是单调的)
    chs = []
    with torch.no_grad():
        h = vae.post_quant_conv(torch.zeros(1, 4, 32, 32, device=DEV))
        h = dec.conv_in(h); h = dec.mid_block(h, None)
        for i, up in enumerate(dec.up_blocks):
            h = up(h, None)
            chs.append(h.shape[1])
    print(f"  up_blocks 输出通道 {chs}")
    heads = nn.ModuleList([nn.Conv2d(c, 3, 1).to(DEV) for c in chs])
    for m in heads:
        nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def decode_trunc(z, stop):
        h = vae.post_quant_conv(z)
        h = dec.conv_in(h)
        h = dec.mid_block(h, None)
        for i, up in enumerate(dec.up_blocks):
            h = up(h, None)
            if i >= stop:
                break
        return heads[stop](h)

    # ── B. 保真度: 截断输出里的骨架还能不能看清 ────────────────────────
    print("=" * 74)
    print("B. 保真度: 截断解码 vs 真骨架 (GT 骨架 latent, 64 条)")
    print("=" * 74)
    import glob
    with np.load(sorted(glob.glob('data/top10_style23/shards_aux_skel3/shard_*.npz'))[0]) as z:
        T = torch.from_numpy(z['latents'][:64].astype(np.float32)).to(DEV)
    ST = generate_binary_structure(2, 2)
    with torch.no_grad():
        full = vae.decode(T / sc).sample
        full = ((full.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)
        gt_sk = torch.stack([torch.from_numpy(skeletonize((a[0].cpu().numpy() < 0.5)))
                             for a in full]).unsqueeze(1).float().to(DEV)
        for stop in (0, 1, 2):
            out = decode_trunc(T / sc, stop)
            g = ((out.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)
            # 阈值取 0.5, 并上采样到 256 比
            m = (g < 0.5).float()
            if m.shape[-1] != 256:
                m = F.interpolate(m, size=(256, 256), mode='nearest')
            iou = float((m.bool() & gt_sk.bool()).sum() / (m.bool() | gt_sk.bool()).sum().clamp_min(1))
            print(f"   stop={stop}  输出 {out.shape[-1]}x{out.shape[-1]}  "
                  f"墨量 {float(m.mean()):.4f} (真骨架 {float(gt_sk.mean()):.4f})  "
                  f"骨架IoU {iou:.4f}")
    print("   (全分辨率 VAE 往返对真骨架 PNG 的 IoU = 0.956, 见 diag_skel_latent_mapping)")

    # ── A. 显存 ────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("A. 显存: deform(B=512) + 截断解码(K, 带梯度) + backward")
    print("=" * 74)
    model = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=128, max_off=6.0,
                       coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1, ckpt=1,
                       stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, topo_mode=1,
                       warp_iters=1, film_mode='film', style_tokens=16, attn_heads=4,
                       preserve_amp=1).to(DEV)
    rng = np.random.RandomState(0)
    G = torch.from_numpy(rng.randn(1024, 4, 32, 32).astype(np.float32)).to(DEV)
    SY = torch.from_numpy(rng.randn(1024, 128).astype(np.float32)).to(DEV)

    def step(B, K, stop):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        g2 = model(G[:B], SY[:B])
        hw = -1
        if K > 0:
            img = decode_trunc(g2[:K] / sc, stop)
            hw = img.shape[-1]
            loss = F.mse_loss(img, torch.zeros_like(img))
        else:
            loss = g2.pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        return (gib(torch.cuda.max_memory_allocated()),
                gib(torch.cuda.max_memory_reserved()), time.time() - t0, hw)

    B = 512
    a0, r0, _, _ = step(B, 0, None)
    print(f"  B={B} 不挂 VAE 基线: alloc {a0:.2f}G resv {r0:.2f}G")
    print(f"\n  {'stop':>5}{'分辨率':>10}{'K':>6}{'峰值alloc':>11}{'resv':>9}"
          f"{'边际':>9}{'sec':>8}{'放得下':>8}")
    print("  " + "-" * 62)
    for stop in (0, 1, 2):
        for K in (16, 32, 64, 128, 256, 512, 1024):
            try:
                a, r, dt, hw = step(B, K, stop)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {stop:>5}{'':>10}{K:>6}{'--':>11}{'OOM':>9}")
                break
            ok = 'OK' if r < TOTAL - 0.6 else '险'
            print(f"  {stop:>5}{hw:>7}x{hw:<3}{K:>6}{a:>10.2f}G{r:>8.2f}G"
                  f"{a-a0:>+8.2f}G{dt:>8.3f}{ok:>8}")
        print()


if __name__ == "__main__":
    main()
