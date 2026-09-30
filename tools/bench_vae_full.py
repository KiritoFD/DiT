#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/bench_vae_full.py — **全部走 VAE decode**: batch 能开到多少 + 梯度透传质量?

⚠ 不开 expandable_segments。

A. 显存: 一个完整训练步 = deform(B) -> vae.decode(整批 B, 带梯度) -> 损失 -> backward。
   扫 6 种配置找最大 B:
     plain / bf16 / ckpt(解码器逐块梯度检查点) / ckpt+bf16 / chunk(逐块解码+即时反传)
     / chunk+ckpt
   ★ chunk 模式是关键: 逐块 decode -> 逐块 backward -> 立刻释放该块图,
     峰值与 B 解耦(只由 chunk 大小决定), 理论上 B 可以任意大。

B. 梯度透传质量 (在可行的 B 上):
   1. g'.grad 是否非空 / 有无 NaN / Inf
   2. 逐通道梯度范数 (是否被某一个通道垄断)
   3. ★ 数值差分校验: 有限差分 vs 解析梯度 的方向余弦 (验证反传链路正确)
   4. 梯度是否**指向真值**: cos(-grad, g_gt - g')
   5. 梯度能量是否只集中在少数像素 (top1% 占比)
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt_fn

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda'
GIB = 2 ** 30
TOTAL = 23.52
SC = 0.18215
CH = 8          # chunk 大小


def gib(x):
    return x / GIB


def main():
    from src.model.deform_skel import DeformSkel
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    for p in vae.parameters():
        p.requires_grad_(False)
    sc = float(getattr(vae.config, 'scaling_factor', SC))
    dec = vae.decoder

    def dec_plain(z):
        return dec(vae.post_quant_conv(z), None)

    def dec_ckpt(z):
        h = vae.post_quant_conv(z)
        h = _ckpt_fn(dec.conv_in, h, use_reentrant=False)
        h = _ckpt_fn(dec.mid_block, h, None, use_reentrant=False)
        for up in dec.up_blocks:
            h = _ckpt_fn(up, h, None, use_reentrant=False)
        h = dec.conv_norm_out(h)
        h = dec.conv_act(h)
        return dec.conv_out(h)

    model = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=128, max_off=6.0,
                       coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1, ckpt=1,
                       stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, topo_mode=1,
                       warp_iters=1, film_mode='film', style_tokens=16, attn_heads=4,
                       preserve_amp=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    rng = np.random.RandomState(0)
    G = torch.from_numpy(rng.randn(2048, 4, 32, 32).astype(np.float32)).to(DEV)
    SY = torch.from_numpy(rng.randn(2048, 128).astype(np.float32)).to(DEV)

    def crit(img):
        """图空间 BCE(pos_weight)+Dice, 目标 = 全白图(骨架 latent 解码 = 白底黑线)。"""
        pr = ((img.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)
        tgt = torch.ones_like(pr)
        logit = (pr - 0.5) * 12.0
        bce = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=torch.tensor(5.0, device=DEV))
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * tgt).sum() + 1e-6) / (p.sum() + tgt.sum() + 1e-6)
        return bce + dice

    def step(B, mode):
        """mode: plain / bf16 / ckpt / ckpt_bf16 / chunk / chunk_ckpt"""
        model.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        g2 = model(G[:B], SY[:B])
        use_bf16 = 'bf16' in mode
        fn = dec_ckpt if 'ckpt' in mode else dec_plain
        if mode.startswith('chunk'):
            for i in range(0, B, CH):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    img = fn(g2[i:i + CH] / sc).float()
                # ⚠ g2 被多个 chunk 共享 -> 除最后一块外必须 retain_graph
                crit(img).backward(retain_graph=(i + CH < B))
        else:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                img = fn(g2 / sc).float()
            crit(img).backward()
        opt.step()
        torch.cuda.synchronize()
        return (gib(torch.cuda.max_memory_allocated()),
                gib(torch.cuda.max_memory_reserved()), time.time() - t0)

    print("=" * 84)
    print("A. 全批走 VAE decode —— 最大 batch")
    print("=" * 84)
    print(f"  {'模式':<14}{'B':>6}{'峰值alloc':>12}{'resv':>10}{'sec':>9}{'放得下':>9}")
    best = {}
    for mode in ('chunk', 'chunk_ckpt', 'chunk_ckpt_bf16'):
        mx = 0
        for B in (16, 32, 64, 128, 256, 512, 1024, 2048):
            try:
                a, r, dt = step(B, mode)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {mode:<14}{B:>6}{'--':>12}{'OOM':>10}")
                break
            ok = 'OK' if r < TOTAL - 0.6 else '险'
            print(f"  {mode:<14}{B:>6}{a:>11.2f}G{r:>9.2f}G{dt:>9.3f}{ok:>9}")
            if ok == 'OK':
                mx = B
        best[mode] = mx
        print()
    print("  最大可行 batch:", {k: v for k, v in best.items()})

    # ── B. 梯度透传质量 ────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("B. 梯度透传质量 (全批 decode, plain 模式, B=16)")
    print("=" * 84)
    B = 16
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    g2 = model(G[:B], SY[:B])
    g2.retain_grad()
    img = dec_plain(g2 / sc).float()
    loss = crit(img)
    loss.backward()
    gr = g2.grad
    print(f"  loss = {float(loss):.5f}")
    print(f"  g'.grad: shape={tuple(gr.shape)}  norm={float(gr.norm()):.6e}  "
          f"NaN={int(torch.isnan(gr).sum())}  Inf={int(torch.isinf(gr).sum())}")
    print(f"  逐通道梯度范数: {[round(float(gr[:, c].norm()), 5) for c in range(4)]}")
    print(f"  逐通道 |grad| 均值: {[round(float(gr[:, c].abs().mean()), 6) for c in range(4)]}")
    # 空间集中度
    e = gr.flatten(1).pow(2)
    top = torch.topk(e, max(1, int(e.shape[1] * 0.01)), dim=1).values.sum(1)
    print(f"  top1% 像素占梯度能量比例: {float((top / e.sum(1)).mean()):.3f} "
          f"(0.01=完全均匀, 1.0=全集中)")
    # 方向是否指向真值 (用 latent 空间的 g_gt 做参考)
    Tg = torch.zeros_like(g2)          # 这里没有真值, 用「图空间目标 = 全白」的 latent 近似
    print("  ⚠ 该处无真值 g_gt, 跳过方向性检查 (见 diag_loss_race 的端到端对比)")

    # 有限差分校验
    print("\n  有限差分校验 (8 个随机 latent 元素, eps=1e-2):")
    g2b = g2.detach().clone()
    idx = [(int(rng.randint(B)), int(rng.randint(4)), int(rng.randint(32)), int(rng.randint(32)))
           for _ in range(8)]

    def loss_at(x):
        # 只验证 VAE 解码这条反传链路: 输入 x 直接喂 decoder
        im = dec_plain(x.detach() / sc).float()
        return crit(im)
    fd_rows = []
    for (b, c, h, w) in idx:
        eps = 1e-2
        xp = g2b.clone(); xp[b, c, h, w] += eps
        xm = g2b.clone(); xm[b, c, h, w] -= eps
        lp = float(loss_at(xp)); lm = float(loss_at(xm))
        fd = (lp - lm) / (2 * eps)
        an = float(gr[b, c, h, w])
        fd_rows.append((fd, an))
    fdv = np.array([r[0] for r in fd_rows]); anv = np.array([r[1] for r in fd_rows])
    print(f"    {'解析梯度':>14}{'有限差分':>14}{'比值':>10}")
    for fd, an in fd_rows:
        print(f"    {an:>14.6e}{fd:>14.6e}{an/max(fd,1e-12):>10.3f}")
    if np.std(anv) > 0 and np.std(fdv) > 0:
        cc = float(np.corrcoef(anv, fdv)[0, 1])
        cos = float((anv @ fdv) / (np.linalg.norm(anv) * np.linalg.norm(fdv) + 1e-12))
        print(f"    相关系数 {cc:.4f}   方向余弦 {cos:.4f}  "
              f"-> {'✓ 反传链路正确' if cos > 0.9 else '✗ 可疑'}")
    else:
        print("    (梯度或差分方差为 0, 无法判定)")


if __name__ == "__main__":
    main()
