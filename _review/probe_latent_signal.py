#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_latent_signal.py — latent 空间里有没有可用信号? 以及为什么 MSE 不行。

三个读数:
  1. 幅度: std / gt 相对背景 z_bg 的能量。若 std 能量 > gt 能量, 则 L2 的
     最优解是"把墨量调小", MSE 测的是幅度而不是几何。
  2. MSE 排序: MSE(gt)=0 < MSE(blank) < MSE(std)? 若是, 说明**画空白比画出输入分数高**,
     L2 的全局最优点在退化解一侧。
  3. 候选损失排序: 把 latent 投影到墨迹方向 -> **质量归一化**(去掉总墨量) -> L1。
     正确损失必须满足 loss(gt) < loss(std) < loss(blank)。
     若通过, 这条损失可以完全在 latent 上算, 不需要 VAE 解码。
"""
import glob
import os
import sys

import numpy as np
import torch as th

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '/root/Workspace/xy/DiT')

Z_BG = th.tensor([2.18129, 1.42018, -0.00979, -1.14073]).view(1, 4, 1, 1)
DELTA = None  # 由 shards 估计


def load(d, n=512):
    fs = sorted(glob.glob(os.path.join(d, 'shard_*.npz')))
    xs, ids = [], []
    for f in fs:
        z = np.load(f)
        xs.append(z['latents'])
        ids.append(z['img_ids'])
        if sum(len(a) for a in xs) >= n:
            break
    X = np.concatenate(xs, 0)[:n]
    I = np.concatenate(ids, 0)[:n]
    return th.from_numpy(X).float(), list(I)


S, si = load('data/top10_style23/shards_std')
T, ti = load('data/top10_style23/shards_aux_skel3')
m = {i: k for k, i in enumerate(si)}
idx = [m[i] for i in ti if i in m]
T = T[[k for k, i in enumerate(ti) if i in m]]
S = S[idx]
print(f'配对样本 {len(idx)}')

# 墨迹方向: 用 GT 与背景的差估计
D = (T - Z_BG).mean(dim=(0, 2, 3), keepdim=True)
D = D / D.norm()
dd = (D * D).sum().clamp_min(1e-9)


def ink(z):
    """投影到墨迹方向 -> (B,1,32,32) 墨迹图 (与原 w_mass 同口径)。"""
    return ((z - Z_BG) * D).sum(1, keepdim=True) / dd


def e(z):
    return float(((z - Z_BG).norm(dim=1)).mean())


BLANK = Z_BG.expand_as(S).contiguous()
print()
print('=== 1. 幅度 (相对背景的能量) ===')
print(f'  std  {e(S):.4f}   gt {e(T):.4f}   blank {e(BLANK):.4f}')
print(f'  -> std/gt 能量比 {e(S) / max(e(T), 1e-9):.2f}')

print()
print('=== 2. 原始 latent MSE 排序 (越小越好) ===')
for nm, z in (('gt(正解)', T), ('blank(空白)', BLANK), ('std(原样)', S),
              ('gt+噪声', T + 0.1 * th.randn_like(T))):
    print(f'  {nm:12s} {float((z - T).pow(2).mean()):.5f}')
print(f'  -> 若 blank < std, L2 的全局最优在退化解一侧')

print()
print('=== 3. 质量归一化墨迹图 L1 (候选 fast loss, 无需 VAE) ===')
it = ink(T)
it_n = it / it.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
for nm, z in (('gt(正解)', T), ('std(原样)', S), ('blank(空白)', BLANK)):
    iz = ink(z)
    iz_n = iz / iz.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    l1 = float((iz_n - it_n).abs().mean())
    print(f'  {nm:12s} 归一化L1 = {l1:.5f}   (未归一化 L1 {float((iz - it).abs().mean()):.5f})')

print()
print('=== 4. 通道标准化后的 L2 (另一种幅度无关量) ===')


def cn(z):
    m_ = z.mean(dim=(2, 3), keepdim=True)
    s_ = z.std(dim=(2, 3), keepdim=True)
    return (z - m_) / (s_ + 1e-5)


ct = cn(T)
for nm, z in (('gt(正解)', T), ('std(原样)', S), ('blank(空白)', BLANK)):
    print(f'  {nm:12s} {float((cn(z) - ct).pow(2).mean()):.5f}')
