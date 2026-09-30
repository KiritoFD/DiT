#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/bench_vae_decode.py — VAE 带梯度解码的显存基准: sub-batch 能开到多少?

⚠ **不使用 expandable_segments**（项目规定：灾难性）。

关键修正（第一版错了）:
  - 第一版预分配了 (4096,3,256,256) 的目标张量 = 3.2GB 常量, 污染所有读数。
  - 第一版没开 `ckpt=1`(梯度检查点), 而真实 runner 是开的。
  - 结论: 必须量**边际成本** = (挂 VAE 的峰值) - (不挂 VAE 的峰值)。
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda'
GIB = 2 ** 30
VAE_PATH = 'data/pretrained/pretrained_models/sd-vae-ft-ema'
TOTAL = 23.52


def gib(x):
    return x / GIB


def main():
    from src.model.deform_skel import DeformSkel
    from diffusers.models import AutoencoderKL

    print(f"GPU total {TOTAL:.2f} GiB | **expandable_segments = OFF** | torch {torch.__version__}")

    def make(ckpt):
        return DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=128, max_off=6.0,
                          coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1,
                          ckpt=ckpt, stroke_mod=1, stroke_cap=1.0, gate_radius=0.25,
                          topo_mode=1, warp_iters=1, film_mode='film', style_tokens=16,
                          attn_heads=4, preserve_amp=1).to(DEV)

    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).eval().to(DEV)
    for p in vae.parameters():
        p.requires_grad_(False)
    sc = float(getattr(vae.config, 'scaling_factor', 0.18215))

    rng = np.random.RandomState(0)
    G = torch.from_numpy(rng.randn(2048, 4, 32, 32).astype(np.float32)).to(DEV)
    ST = torch.from_numpy(rng.randn(2048, 128).astype(np.float32)).to(DEV)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"常驻(VAE+G/ST) {gib(torch.cuda.memory_allocated()):.2f} GiB")

    def step(model, B, K, mode='off'):
        """返回 (peak_alloc, peak_resv, sec)。K=0 = 不挂 VAE。"""
        if mode == 'slicing':
            vae.enable_slicing()
        elif mode == 'tiling':
            vae.enable_tiling()
        else:
            for f in ('disable_slicing', 'disable_tiling'):
                if hasattr(vae, f):
                    getattr(vae, f)()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        g2 = model(G[:B], ST[:B])
        if K > 0:
            img = vae.decode(g2[:K] / sc).sample
            tgt = torch.zeros_like(img)                    # ★ 不预分配大张量
            loss = F.mse_loss(img, tgt)
        else:
            loss = g2.pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        return (gib(torch.cuda.max_memory_allocated()),
                gib(torch.cuda.max_memory_reserved()), time.time() - t0)

    def safe(model, B, K, mode='off'):
        try:
            return step(model, B, K, mode)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, None, None

    for ck in (1, 0):
        print(f"\n{'='*74}\nckpt(梯度检查点)={ck}\n{'='*74}")
        model = make(ck)
        # 基线: 不挂 VAE
        base = {}
        for B in (256, 512, 1024, 2048):
            a, r, dt = safe(model, B, 0)
            if a is None:
                print(f"  B={B:<5} K=0       --          OOM")
            else:
                base[B] = a
                print(f"  B={B:<5} K=0   峰值 alloc {a:>6.2f}G  resv {r:>6.2f}G  {dt:.3f}s")
        print(f"  {'-'*70}")
        print(f"  {'B':>5}{'K':>5}{'模式':>10}{'峰值alloc':>11}{'resv':>9}"
              f"{'VAE边际':>10}{'step(s)':>9}{'放得下':>8}")
        for B in (256, 512, 1024):
            if B not in base:
                continue
            for K in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
                a, r, dt = safe(model, B, K)
                if a is None:
                    print(f"  {B:>5}{K:>5}{'off':>10}{'--':>11}{'OOM':>9}")
                    break                                    # 后面只会更差, 停
                ok = 'OK' if r < TOTAL - 0.6 else '险'
                print(f"  {B:>5}{K:>5}{'off':>10}{a:>10.2f}G{r:>8.2f}G"
                      f"{a-base[B]:>+9.2f}G{dt:>9.3f}{ok:>8}")
        # 省显存开关 (只测 B=512)
        for mode in ('slicing', 'tiling'):
            if 512 not in base:
                continue
            print(f"  --- {mode} (B=512) ---")
            for K in (32, 64, 128, 256, 512):
                a, r, dt = safe(model, 512, K, mode)
                if a is None:
                    print(f"       K={K:<5} OOM")
                    break
                print(f"       K={K:<5} alloc {a:>6.2f}G  resv {r:>6.2f}G  "
                      f"边际 {a-base[512]:>+5.2f}G  {dt:.3f}s")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
