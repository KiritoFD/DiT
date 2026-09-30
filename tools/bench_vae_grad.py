#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/bench_vae_grad.py — VAE 解码这条反传链路: 梯度能不能良好透传?

用真实数据:
    g_gt  = 真骨架 latent
    tgt   = decode(g_gt)                       (no_grad, 目标图)
    g'    = g_gt 加扰动 (模拟"还没训好的输出")
    loss  = BCE(pos_weight) + Dice( decode(g'), tgt )
    loss.backward()
检查:
  1. grad 存在 / 无 NaN / Inf / 非零
  2. 逐通道梯度范数 (是否被单通道垄断 -> 病态)
  3. 有限差分校验 (解析梯度 vs 数值梯度, 方向余弦)
  4. ★ 梯度是否指向真值: cos(-grad, g_gt - g')
  5. 梯度能量空间集中度
  6. 每样本梯度 (batch 内是否有样本完全没有梯度)
显存: 用 chunk + 梯度检查点 (见 bench_vae_full: B=512 -> 13.95G 可行)。
"""
import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ck

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda'
GIB = 2 ** 30


def main():
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    for p in vae.parameters():
        p.requires_grad_(False)
    sc = float(getattr(vae.config, 'scaling_factor', 0.18215))
    dec = vae.decoder

    def dec_ck(z):
        h = vae.post_quant_conv(z)
        h = _ck(dec.conv_in, h, use_reentrant=False)
        h = _ck(dec.mid_block, h, None, use_reentrant=False)
        for up in dec.up_blocks:
            h = _ck(up, h, None, use_reentrant=False)
        h = dec.conv_norm_out(h)
        h = dec.conv_act(h)
        return dec.conv_out(h)

    @torch.no_grad()
    def dec_nograd(z, ch=16):
        outs = []
        for i in range(0, z.shape[0], ch):
            outs.append(vae.decode(z[i:i + ch] / sc).sample.float())
        return torch.cat(outs, 0)

    with np.load(sorted(glob.glob('data/top10_style23/shards_aux_skel3/shard_*.npz'))[0]) as z:
        T = torch.from_numpy(z['latents'][:8].astype(np.float32)).to(DEV)
    B = T.shape[0]
    tgt = dec_nograd(T)                       # (B,3,256,256) in [-1,1]
    tg01 = ((tgt.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)

    def loss_fn(img, target01):
        pr = ((img.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)
        logit = (pr - 0.5) * 12.0
        pos = target01.sum().clamp_min(1.0)
        pw = ((target01.numel() - pos) / pos).clamp(1.0, 10.0)
        bce = F.binary_cross_entropy_with_logits(logit, target01.expand_as(logit),
                                                pos_weight=pw)
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * target01).sum() + 1e-6) / (p.sum() + target01.sum() + 1e-6)
        return bce + dice

    print("=" * 84)
    print("1/2. 梯度存在性 / 逐通道分布  (B=8, chunk+梯度检查点解码)")
    print("=" * 84)
    for name, g0 in [('干净 g_gt (loss 应在最低点)', T.clone()),
                     ('扰动 g_gt + 0.3*noise (模拟未训好)', None)]:
        if g0 is None:
            g0 = T.clone() + 0.3 * torch.randn_like(T)
        x = g0.detach().clone().requires_grad_(True)
        img = dec_ck(x / sc)
        loss = loss_fn(img, tg01)
        loss.backward()
        gr = x.grad
        print(f"\n  [{name}]  loss={float(loss):.5f}")
        print(f"    grad: shape={tuple(gr.shape)} norm={float(gr.norm()):.5e} "
              f"NaN={int(torch.isnan(gr).sum())} Inf={int(torch.isinf(gr).sum())}")
        print(f"    逐通道梯度范数 : {[round(float(gr[:, c].norm()), 5) for c in range(4)]}")
        print(f"    逐通道 |grad|均 : {[round(float(gr[:, c].abs().mean()), 6) for c in range(4)]}")
        e = gr.flatten(1).pow(2)
        top = torch.topk(e, max(1, int(e.shape[1] * 0.01)), dim=1).values.sum(1)
        print(f"    top1% 像素占能量: {float((top / e.sum(1)).mean()):.3f} (0.01=均匀)")
        per = gr.flatten(1).norm(dim=1)
        print(f"    每样本梯度范数  : {[round(float(v), 5) for v in per]}  "
              f"(min/max = {float(per.min()/per.max()):.3f})")
        # 方向性
        d = (T - g0)
        cos = float(F.cosine_similarity((-gr).flatten(), d.flatten(), dim=0))
        print(f"    ★ cos(-grad, g_gt - g') = {cos:+.4f}   "
              f"-> {'✓ 指向真值' if cos > 0.05 else '✗ 不指向真值'}")
        del x, img, loss, gr
        torch.cuda.empty_cache()

    print("\n" + "=" * 84)
    print("3. 有限差分校验 (解析梯度 vs 数值梯度)")
    print("=" * 84)
    x0 = T.clone()
    x0.requires_grad_(True)
    img = dec_ck(x0 / sc)
    loss = loss_fn(img, tg01)
    loss.backward()
    gr = x0.grad.detach().clone()
    rng = np.random.RandomState(0)
    idx = [(int(rng.randint(B)), int(rng.randint(4)), int(rng.randint(32)),
            int(rng.randint(32))) for _ in range(6)]
    rows = []
    for (b, c, h, w) in idx:
        eps = 5e-3
        vals = []
        for s in (+eps, -eps):
            xx = T.clone()
            xx[b, c, h, w] += s
            with torch.no_grad():
                vals.append(float(loss_fn(dec_nograd(xx), tg01)))
        fd = (vals[0] - vals[1]) / (2 * eps)
        rows.append((float(gr[b, c, h, w]), fd))
    print(f"    {'解析梯度':>15}{'有限差分':>15}{'比值':>10}")
    for an, fd in rows:
        print(f"    {an:>15.6e}{fd:>15.6e}{an/max(abs(fd),1e-12):>10.3f}")
    anv = np.array([r[0] for r in rows]); fdv = np.array([r[1] for r in rows])
    if np.std(anv) > 0 and np.std(fdv) > 0:
        cc = float(np.corrcoef(anv, fdv)[0, 1])
        cs = float((anv @ fdv) / (np.linalg.norm(anv) * np.linalg.norm(fdv) + 1e-12))
        print(f"    相关系数 {cc:+.4f}  方向余弦 {cs:+.4f}  "
              f"-> {'✓ 反传链路正确' if cs > 0.9 else '⚠ 需人工确认'}")
    else:
        print("    (方差为 0, 无法判定)")


if __name__ == "__main__":
    main()
