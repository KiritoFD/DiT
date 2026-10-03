#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GlyphQuery 验证 v2（修正 v1 的三个测试 bug）。

v1 的 bug：
  · 测试3 直接给 zero-init 的 style 权重乘倍数 -> 0×1000 还是 0, 等于没放大
  · 测试5 没打破 out_proj 的 zero-init -> 输出恒 0, "keep=1 应有变化"必然失败
  · 测试6 对 nn.Linear 取 .grad（应取 .weight.grad）
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from src.model.glyph_query import GlyphQuery  # noqa: E402

torch.manual_seed(0)
D, CD, H, WIN, GRID = 384, 128, 4, 5, 16
B, N = 8, GRID * GRID
hd = D // H
print(f'head_dim={hd}  理论 logit 上界 ≈ sqrt(hd) = {hd ** 0.5:.2f}\n')


def fresh(trained=True):
    """trained=True 时把 zero-init 打开，模拟训练若干步之后的状态。"""
    m = GlyphQuery(D, CD, num_heads=H, rank=64, window=WIN, grid_size=GRID)
    if trained:
        with torch.no_grad():
            nn.init.normal_(m.out_proj.weight, std=0.02)
            nn.init.normal_(m.style_to_q.weight, std=0.02)
            nn.init.normal_(m.style_to_k.weight, std=0.02)
    return m


x = torch.randn(B, N, D)
g = torch.randn(B, N, D)
e = torch.randn(B, CD)
pos = torch.randn(1, N, D)

# ── 1. 形状 / NaN ───────────────────────────────────────────────────
m = fresh()
out = m(x, g, e, pos)
print(f'[1] out.shape={tuple(out.shape)} NaN={bool(torch.isnan(out).any())}')
print(f'[2] exp(out_log_scale) 初值 = {float(fresh(False).out_log_scale.exp()):.6f}')

# ── 3. ★ 核心主张 ───────────────────────────────────────────────────
print('\n[3] ★ 风格投影放大后 logit 是否有界')


def logits(module, scale_style, qknorm):
    with torch.no_grad():
        sq, sk = module.style_to_q, module.style_to_k
        q = module.q_proj(module.norm_q(x + pos)) + scale_style * sq(e).unsqueeze(1)
        k = module.k_proj(module.norm_kv(g + pos)) + scale_style * sk(e).unsqueeze(1)
        q = q.view(B, N, H, hd).transpose(1, 2)
        k = k.view(B, N, H, hd).transpose(1, 2)
        if qknorm:
            q, k = F.rms_norm(q, (hd,)), F.rms_norm(k, (hd,))
        s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
        if module.win_mask is not None:
            s = s.masked_fill(~module.win_mask.to(torch.bool),
                              torch.finfo(s.dtype).min)
        real = s[s > torch.finfo(s.dtype).min / 2]
        return float(real.abs().max()), bool(torch.isnan(s).any())


mt = fresh()
print(f'  {"风格放大":>8} | {"有 QK-Norm":>18} | {"无 QK-Norm (对照)":>22}')
print('  ' + '-' * 56)
for sc in [1, 10, 100, 1000, 10000]:
    a, na = logits(mt, sc, True)
    b, nb = logits(mt, sc, False)
    print(f'  {sc:>8}x | {a:8.2f} NaN={str(na):5s} | {b:14.2f} NaN={str(nb):5s}')

# ── 4. 窗口 mask ────────────────────────────────────────────────────
print('\n[4] 窗口 mask')
with torch.no_grad():
    q = mt.q_proj(mt.norm_q(x + pos)) + mt.style_to_q(e).unsqueeze(1)
    k = mt.k_proj(mt.norm_kv(g + pos)) + mt.style_to_k(e).unsqueeze(1)
    q = F.rms_norm(q.view(B, N, H, hd).transpose(1, 2), (hd,))
    k = F.rms_norm(k.view(B, N, H, hd).transpose(1, 2), (hd,))
    s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
    s = s.masked_fill(~mt.win_mask.to(torch.bool), torch.finfo(s.dtype).min)
    attn = torch.softmax(s, dim=-1)
    print(f'  窗口外 attention max = {float(attn[~mt.win_mask.expand_as(attn)].abs().max()):.3e}')
    print(f'  每行 attention 和 = {float(attn.sum(-1).mean()):.6f}')
    print(f'  每行参与 token 数 = {float((attn > 0).float().sum(-1).mean()):.1f} (窗口 5x5=25)')

# ── 5. keep 语义（这次先打破 zero-init）──────────────────────────────
print('\n[5] keep=0 的样本输出是否为 0')
keep = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1], dtype=torch.bool)
o2 = mt(x, g, e, pos, keep=keep)
d = (o2 - x).abs().amax(dim=(1, 2))
print(f'  keep=0: max|Δx| = {float(d[~keep].max()):.3e} (期望 0)')
print(f'  keep=1: max|Δx| = {float(d[keep].min()):.3e} (期望 >0)')

# ── 6. 梯度 ─────────────────────────────────────────────────────────
print('\n[6] 梯度能否流动')
xx = torch.randn(B, N, D, requires_grad=True)
mt(xx, g, e, pos).pow(2).mean().backward()
for nm, p in [('q_proj', mt.q_proj.weight), ('k_proj', mt.k_proj.weight),
              ('v_proj', mt.v_proj.weight), ('out_proj', mt.out_proj.weight),
              ('style_to_q', mt.style_to_q.weight),
              ('style_to_k', mt.style_to_k.weight),
              ('out_log_scale', mt.out_log_scale)]:
    print(f'  {nm:>15} grad.norm = {float(p.grad.norm()):.3e}')
print(f'  {"input x":>15} grad.norm = {float(xx.grad.norm()):.3e}')

# ── 7. 零初始化下第 0 步是否恒等（可 resume 的前提）──────────────────
print('\n[7] 零初始化下 step0 是否恒等')
mz = fresh(trained=False)
oz = mz(x, g, e, pos)
print(f'  |out - x| max = {float((oz - x).abs().max()):.3e} (期望 0)')

# ── 8. bf16 下是否稳定（训练用 autocast bf16）───────────────────────
print('\n[8] bf16 下是否稳定')
with torch.autocast('cpu', dtype=torch.bfloat16):
    ob = mt(x, g, e, pos)
print(f'  bf16 out dtype={ob.dtype} NaN={bool(torch.isnan(ob.float()).any())} '
      f'inf={bool(torch.isinf(ob.float()).any())}')
