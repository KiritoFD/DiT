#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkelNet 够不够用？—— 算「不可约下界」来判定。

## 为什么需要一个下界
头的输入是 (g_std, 书家)。但 **g_std 是按字共享的**（实测 43.5% 的字只有 1 张 std），
而同一个 (字, 书家) 往往有**多个样本**，它们的 g_gt 各不相同。
-> 头对这些样本只能输出**同一个** g' -> 最多只能预测组内均值
-> 残差有一个**不可约下界 = 组内方差**

所以判据不是"残差多小算好"，而是：
  **残差 / 下界** —— 接近 1.0 说明已经到极限，加训练/加容量都没用；
  明显 >1 说明还有空间。

## 三个问题的答案都在这
  · 训练充分吗   -> 残差是否已贴近下界
  · 参数合适吗   -> 同上（贴着下界就说明容量够）
  · 够下游用吗   -> 看"闭合率"相对**可达上限**是多少

用法: python tools/diag_skeln et_floor.py --ckpt assets/deform_skel_v5.pt
"""
import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np
import torch

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.model.deform_skel import DeformSkel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='assets/deform_skel_v5.pt')
ap.add_argument('--std-dir', default='data/50k/shards_std_fixed')
ap.add_argument('--gt-dir', default='data/50k/shards_aux_skel3')
ap.add_argument('--width', type=int, default=96)
ap.add_argument('--batch', type=int, default=2048)
a = ap.parse_args()

DEV = 'cpu'          # 一次性诊断, 用 CPU 不占 GPU


def load_bank(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


A, B = load_bank(a.std_dir), load_bank(a.gt_dir)
rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
C2I = {}
for r in rows:
    C2I.setdefault(str(r.get('calligrapher', '')), len(C2I))
meta = {}
for r in rows:
    try:
        i = int(os.path.basename(r.get('image_path', '')).split('.')[0])
    except Exception:
        continue
    if i in A and i in B:
        meta[i] = (C2I.get(str(r.get('calligrapher', '')), 0), str(r.get('character', '')))
ids = sorted(meta)
G = np.stack([A[i] for i in ids])
T = np.stack([B[i] for i in ids])
Y = np.array([meta[i][0] for i in ids])
CH = [meta[i][1] for i in ids]
print(f'[1] 样本 {len(ids)}')

# ── 头的输出 ──
sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
head = DeformSkel(cond_dim=128, ch=4, grid=32, residual=1, width=a.width,
                  max_off=6.0, res_cap=2.0).eval()
head.load_state_dict(sd['deform'], strict=True)
tab = torch.load('assets/callig_emb_pretrained_50k.pt', map_location='cpu',
                 weights_only=False)['embedding'].float()
out = np.empty_like(T, dtype=np.float32)
with torch.no_grad():
    for s in range(0, len(ids), a.batch):
        j = min(s + a.batch, len(ids))
        out[s:j] = head(torch.from_numpy(G[s:j]),
                        tab[torch.from_numpy(Y[s:j])]).numpy()
print(f'[2] 头输出完成')

# ── 各口径的 MSE ──
mse_base = float(((G - T) ** 2).mean())        # 什么都不做
mse_head = float(((out - T) ** 2).mean())      # 头
print()
print('=== MSE ===')
print(f'  基线 MSE(g_std, g_gt)        = {mse_base:.5f}')
print(f'  头     MSE(g\', g_gt)          = {mse_head:.5f}   闭合率 {100*(1-mse_head/mse_base):.1f}%')

# ── 不可约下界: 同 (字, 书家) 组内的 g_gt 方差 ──
grp = collections.defaultdict(list)
for k in range(len(ids)):
    grp[(CH[k], int(Y[k]))].append(k)
# 也按"只按字"分组（如果书家条件无效，头最多只能预测按字的均值）
grp_char = collections.defaultdict(list)
for k in range(len(ids)):
    grp_char[CH[k]].append(k)


def within_var(groups, n):
    """组内方差（按样本数加权）: 若头只输出该组均值, 残差就是这个值。

    ⚠ 必须是 .mean() 而不是 .sum(): 用 .sum() 会把元素数(4*32*32)也乘进去,
      算出来的"下界"比 MSE 尺度大三个量级(实测 438.9 vs 0.5) —— 第一版就是这么错的,
      还据此得出了"已贴近下界"的错误结论。
    """
    tot = 0.0
    for v in groups.values():
        if len(v) < 2:
            continue
        arr = T[v]
        tot += float(((arr - arr.mean(0)) ** 2).mean()) * len(v)
    return tot / n


floor_cc = within_var(grp, len(ids))            # 同字同书家的组内方差
floor_c = within_var(grp_char, len(ids))        # 同字的组内方差
n_multi = sum(1 for v in grp.values() if len(v) >= 2)
n_samp = sum(len(v) for v in grp.values() if len(v) >= 2)
print()
print('=== 不可约下界（头最多只能预测条件均值，残差至少这么大）===')
print(f'  同(字,书家) 分组: {len(grp)} 组, 其中 {n_multi} 组有多样本（共 {n_samp} 条）')
print(f'  下界·同字同书家 = {floor_cc:.5f}')
print(f'  下界·仅同字     = {floor_c:.5f}')
print()
print('=== 判定 ===')
r1 = mse_head / max(floor_cc, 1e-9)
r2 = mse_head / max(floor_c, 1e-9)
print(f'  残差 / 下界(同字同书家) = {r1:.2f}   <- 接近 1.0 = 已到极限')
print(f'  残差 / 下界(仅同字)     = {r2:.2f}')
reach = 1 - floor_cc / mse_base
print(f'  可达上限（闭合率的天花板）  = {100*reach:.1f}%')
print(f'  头实际闭合率                = {100*(1-mse_head/mse_base):.1f}%')
print(f'  -> 已达到可达上限的 {100*(1-mse_head/mse_base)/max(reach,1e-9):.0f}%')
print()
if r1 < 1.25:
    print('  结论: **已贴近信息论下界** -> 预训练充分、容量足够，加训练/加参数都没用。')
    print('        下游拿到的已经是"仅凭 (g_std, 书家) 能给出的最好骨架"。')
elif r1 < 2.0:
    print('  结论: 距下界还有空间(1.25~2x) -> 加容量/加步数可能还有收益。')
else:
    print('  结论: 距下界很远(>2x) -> 模型或训练有问题, 值得重做。')
