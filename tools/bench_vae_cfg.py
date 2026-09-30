#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/bench_vae_cfg.py — 目标配置: 总 batch 64, 解码 chunk 16, **不开解码器梯度检查点**。

⚠ 关键纠正: 先前的"分块"写法把 g2 一次前向、再分块 backward(retain_graph=True) ——
   retain_graph 会让**解码器激活不被释放**, 分块等于白分, 所以全 OOM。
   正确做法: **每个 chunk 独立前向 + 立即 backward(不 retain)** = 梯度累加。
   代价是 deform 前向做 N 次; 收益是峰值 = deform(CH) + decoder(CH), 与总 batch 解耦。

测两种真实写法:
  A. 纯累加:      for ch: g2c=deform(ch); decode; loss.backward()   (no retain)
  B. 两遍:        pass1 全批 latent+对比损失 backward; pass2 分块像素 backward
                  (对比损失需要全批, 所以必须两遍; 峰值 = max(deform(B), deform(CH)+dec(ch)))
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
TOTAL = 23.52


def main():
    from src.model.deform_skel import DeformSkel
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    for p in vae.parameters():
        p.requires_grad_(False)
    sc = float(getattr(vae.config, 'scaling_factor', 0.18215))

    def crit(img):
        pr = ((img.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)
        tgt = torch.ones_like(pr)
        logit = (pr - 0.5) * 12.0
        bce = F.binary_cross_entropy_with_logits(
            logit, tgt, pos_weight=torch.tensor(5.0, device=DEV))
        p = torch.sigmoid(logit)
        return bce + (1 - (2 * (p * tgt).sum() + 1e-6) / (p.sum() + tgt.sum() + 1e-6))

    rng = np.random.RandomState(0)
    G = torch.from_numpy(rng.randn(512, 4, 32, 32).astype(np.float32)).to(DEV)
    SY = torch.from_numpy(rng.randn(512, 128).astype(np.float32)).to(DEV)
    TG = torch.from_numpy(rng.randn(512, 4, 32, 32).astype(np.float32)).to(DEV)

    print(f"{'写法':>5}{'deformCk':>9}{'bf16':>6}{'B':>5}{'CH':>5}"
          f"{'peak alloc':>12}{'resv':>9}{'sec':>8}{'放得下':>9}")
    print("  " + "-" * 72)

    for dck in (1, 0):
        m = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=128, max_off=6.0,
                       coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1, ckpt=dck,
                       stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, topo_mode=1,
                       warp_iters=1, film_mode='film', style_tokens=16, attn_heads=4,
                       preserve_amp=1).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def run(tag, B, CH, bf, two_pass):
            try:
                m.zero_grad(set_to_none=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.time()
                nch = B // CH
                if two_pass:
                    g2 = m(G[:B], SY[:B])
                    F.mse_loss(g2, TG[:B]).backward()
                    del g2
                for i in range(0, B, CH):
                    g2c = m(G[i:i + CH], SY[i:i + CH])
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf):
                        img = vae.decode(g2c / sc).sample.float()
                    (crit(img) / nch).backward()
                opt.step()
                torch.cuda.synchronize()
                a = torch.cuda.max_memory_allocated() / GIB
                r = torch.cuda.max_memory_reserved() / GIB
                ok = 'OK' if r < TOTAL - 0.6 else '险'
                print(f"  {tag:>5}{dck:>9}{str(bf):>6}{B:>5}{CH:>5}{a:>11.2f}G"
                      f"{r:>8.2f}G{time.time()-t0:>8.2f}{ok:>9}")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {tag:>5}{dck:>9}{str(bf):>6}{B:>5}{CH:>5}{'--':>12}{'OOM':>9}")

        for bf in (False, True):
            for B, CH in ((64, 16), (64, 8), (128, 16)):
                run('A', B, CH, bf, False)
            run('B', 64, 16, bf, True)
            run('B', 128, 16, bf, True)
            print()
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
