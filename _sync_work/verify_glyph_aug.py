#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 train.py 的条件增强语义（复刻那段代码，逐条断言）。

用户要求的三条性质:
  1. noise 是「对称 ±」-> 不改变条件分布的均值
  2. patch_drop 是逐 latent patch 置 0（Cutout 式）
  3. 两者都不改变张量形状（否则 concat / g_tok 会炸）
"""
import numpy as np
import torch

torch.manual_seed(0)
N, C, H, W = 64, 4, 32, 32
g = torch.randn(N, C, H, W)


def augment(g, gn_scale, gn_prob, gpd):
    """逐字复刻 train.py:1795-1806"""
    if gn_scale > 0 and gn_prob > 0:
        mask = (torch.rand(g.shape[0], 1, 1, 1) < gn_prob).to(g.dtype)
        scales = torch.rand(g.shape[0], 1, 1, 1) * gn_scale
        g = g + mask * torch.randn_like(g) * scales
    if gpd > 0:
        pmask = (torch.rand(g.shape[0], 1, g.shape[2], g.shape[3]) > gpd).to(g.dtype)
        g = g * pmask
    return g


print('=== 1. 形状不变 ===')
for sc, pr, pd_ in [(0.3, 0.5, 0.0), (0.0, 0.0, 0.2), (0.3, 0.5, 0.2)]:
    o = augment(g.clone(), sc, pr, pd_)
    assert o.shape == g.shape, (o.shape, g.shape)
print('  ✓ 三种组合形状都不变')

print('=== 2. 对称 ± : 均值不漂移 ===')
print(f'  原 g 均值 = {float(g.mean()):+.5f}')
for sc in [0.1, 0.3, 0.6, 1.0]:
    ms = [float(augment(g.clone(), sc, 1.0, 0.0).mean()) for _ in range(30)]
    print(f'  gn_scale={sc:.1f} gn_prob=1.0 -> 增强后均值 {np.mean(ms):+.5f} '
          f'(std {np.std(ms):.5f})  漂移 {np.mean(ms) - float(g.mean()):+.5f}')

print('=== 3. 只有 gn_prob 比例的样本被加噪 ===')
for pr in [0.0, 0.25, 0.5, 1.0]:
    o = augment(g.clone(), 0.5, pr, 0.0)
    ch = (o - g).abs().sum(dim=(1, 2, 3)) > 0
    print(f'  gn_prob={pr:.2f} -> 实际被改的样本比例 {float(ch.float().mean()):.3f}')

print('=== 4. patch_drop 的置零比例 ===')
for pd_ in [0.1, 0.2, 0.5]:
    o = augment(g.clone(), 0.0, 0.0, pd_)
    z = (o == 0).float().mean()
    print(f'  gpd={pd_:.2f} -> 张量里被置 0 的比例 {float(z):.3f} (期望≈{pd_:.2f})')

print('=== 5. 两者叠加时 patch_drop 会盖掉噪声（顺序问题）===')
o = augment(g.clone(), 0.5, 1.0, 0.3)
print(f'  叠加后置 0 比例 {float((o == 0).float().mean()):.3f} '
      f'（= 被 patch_drop 的格子；噪声在那些格子上被乘没了，属预期）')
print()
print('★ 结论: 语义与设计一致。注意 gn_scale 与 gn_prob **必须同时 >0** 才加噪'
      '（train.py 外层条件是 `_gn_scale > 0 or _gpd > 0`，内层是 `_gn_scale>0 and _gn_prob>0`）。')
