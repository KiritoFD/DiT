#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GlyphQuery 验证：设计主张是否成立。

主张（用户方案的核心理由）：
  「风格只改变 Q/K 的方向，长度由 QK-RMSNorm 吃掉 -> 风格投影可以变大，logit 仍在 ~sqrt(hd)」
  「不用 clamp；截断处梯度为 0，上一轮就是在 softmax 之前这样炸的」

逐条测：
  1. 形状 / dtype / 无 NaN
  2. out_log_scale 初值 = 0.1
  3. ★ 把 style_to_q / style_to_k 权重放大 1x/10x/100x/1000x，看 max|logit| 是否仍有界
     （对照：把 QK-RMSNorm 关掉，同样的放大立刻爆）
  4. 窗口 mask 是否真的生效（越界位置的 attention 是否为 0）
  5. keep=0 的样本输出是否为 0（drop 语义保护）
  6. 梯度是否能流到 q/k/v 与 style 投影（不是 0）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cpu'
D, CD, H, WIN, GRID = 384, 128, 4, 5, 16
B, N = 8, GRID * GRID

import sys
sys.path.insert(0, '.')
from src.model.glyph_query import GlyphQuery  # noqa: E402

m = GlyphQuery(D, CD, num_heads=H, rank=64, window=WIN, grid_size=GRID).to(DEV)
hd = m.head_dim
print(f'head_dim={hd}  理论 logit 上界 ≈ sqrt(hd) = {hd ** 0.5:.2f}')

x = torch.randn(B, N, D)
g = torch.randn(B, N, D)
e = torch.randn(B, CD)
pos = torch.randn(1, N, D)

print('\n=== 1. 形状 / NaN ===')
out = m(x, g, e, pos)
print(f'  out.shape={tuple(out.shape)} (期望 {(B, N, D)})  NaN={bool(torch.isnan(out).any())}')

print('\n=== 2. out_log_scale 初值 ===')
print(f'  exp(out_log_scale) = {float(m.out_log_scale.exp()):.6f} (期望 0.1)')

print('\n=== 3. ★ 核心主张: 风格投影放大后 logit 是否有界 ===')


def max_logit(module, scale_style, disable_qknorm=False):
    """复刻 forward 的 logit 计算，可选放大风格权重 / 关掉 QK 归一化。"""
    with torch.no_grad():
        sq, sk = module.style_to_q, module.style_to_k
        wq, wk = sq.weight.clone(), sk.weight.clone()
        bq, bk = sq.bias.clone(), sk.bias.clone()
        sq.weight.mul_(scale_style)
        sq.bias.mul_(scale_style)
        sk.weight.mul_(scale_style)
        sk.bias.mul_(scale_style)
        try:
            q = module.q_proj(module.norm_q(x + pos)) + sq(e).unsqueeze(1)
            k = module.k_proj(module.norm_kv(g + pos)) + sk(e).unsqueeze(1)
            q = q.view(B, N, H, hd).transpose(1, 2)
            k = k.view(B, N, H, hd).transpose(1, 2)
            if not disable_qknorm:
                q = F.rms_norm(q, (hd,))
                k = F.rms_norm(k, (hd,))
            s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
            if module.win_mask is not None:
                s = s.masked_fill(~module.win_mask.to(torch.bool),
                                  torch.finfo(s.dtype).min)
            real = s[s > torch.finfo(s.dtype).min / 2]
            return float(real.abs().max())
        finally:
            sq.weight.copy_(wq); sq.bias.copy_(bq)
            sk.weight.copy_(wk); sk.bias.copy_(bk)


print(f'  {"风格放大":>10} | {"有 QK-Norm":>12} | {"无 QK-Norm(对照)":>16}')
print('  ' + '-' * 46)
for sc in [1, 10, 100, 1000]:
    a = max_logit(m, sc, disable_qknorm=False)
    b = max_logit(m, sc, disable_qknorm=True)
    print(f'  {sc:>10}x | {a:12.2f} | {b:16.2f}')

print('\n=== 4. 窗口 mask 是否生效 ===')
with torch.no_grad():
    q = m.q_proj(m.norm_q(x + pos)) + m.style_to_q(e).unsqueeze(1)
    k = m.k_proj(m.norm_kv(g + pos)) + m.style_to_k(e).unsqueeze(1)
    q = F.rms_norm(q.view(B, N, H, hd).transpose(1, 2), (hd,))
    k = F.rms_norm(k.view(B, N, H, hd).transpose(1, 2), (hd,))
    s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
    s = s.masked_fill(~m.win_mask.to(torch.bool), torch.finfo(s.dtype).min)
    attn = torch.softmax(s, dim=-1)
    outside = attn[~m.win_mask.expand_as(attn)]
    print(f'  窗口外 attention 的 max = {float(outside.abs().max()):.3e} (期望 0)')
    print(f'  每行 attention 和 = {float(attn.sum(-1).mean()):.6f} (期望 1)')
    print(f'  每行参与 token 数 = {float((attn > 0).float().sum(-1).mean()):.1f} '
          f'(窗口 {WIN}x{WIN} 的邻域)')

print('\n=== 5. keep=0 的样本输出是否为 0（drop 语义保护）===')
keep = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1], dtype=torch.bool)
o2 = m(x, g, e, pos, keep=keep)
delta = o2 - x
print(f'  keep=0 的样本 Δx 的 max|.| = {float(delta[~keep].abs().max()):.3e} (期望 0)')
print(f'  keep=1 的样本 Δx 的 max|.| = {float(delta[keep].abs().max()):.3e} (期望 >0)')

print('\n=== 6. 梯度能否流动 ===')
m2 = GlyphQuery(D, CD, num_heads=H, rank=64, window=WIN, grid_size=GRID).to(DEV)
# 打破 zero-init，模拟训练若干步之后的状态
with torch.no_grad():
    nn.init.normal_(m2.out_proj.weight, std=0.02)
    nn.init.normal_(m2.style_to_q.weight, std=0.02)
    nn.init.normal_(m2.style_to_k.weight, std=0.02)
    m2.out_log_scale.data.fill_(-2.302585)
xx = torch.randn(B, N, D, requires_grad=True)
oo = m2(xx, g, e, pos)
oo.pow(2).mean().backward()
for nm, p in [('q_proj', m2.q_proj), ('k_proj', m2.k_proj), ('v_proj', m2.v_proj),
              ('out_proj', m2.out_proj), ('style_to_q', m2.style_to_q),
              ('style_to_k', m2.style_to_k), ('out_log_scale', m2.out_log_scale)]:
    gn = float(p.grad.norm()) if p.grad is not None else float('nan')
    print(f'  {nm:>15} grad.norm = {gn:.3e}')
print(f'  {"input x":>15} grad.norm = {float(xx.grad.norm()):.3e}')
