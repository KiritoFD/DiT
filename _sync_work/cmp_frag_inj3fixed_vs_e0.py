#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""断笔比同 step 对比 —— 严格用「229 条未改标签样本」, 绕开旧集/修复集的口径断裂。

E0 的图是旧评测集下生成的, inj3-fixed 的是修复集下生成的; 那 20 条改标签样本
两边的 g 不同 -> 不可比。只用 idx 不在改动列表里的 229 条。
"""
import csv
import os

import numpy as np

OLD = 'assets/eval_v13_strict.csv'
NEW = 'assets/eval_v13_strict_fixed.csv'

E0 = 'assets/results/v17_s2_s2z_baseline/eval_backfill_frag_per_sample.csv'
FX = 'assets/results/v17_inj3_fixed_100k/eval_backfill_frag_per_sample.csv'


def changed_idx():
    a = list(csv.DictReader(open(OLD, encoding='utf-8')))
    b = list(csv.DictReader(open(NEW, encoding='utf-8')))
    return {i for i, (ra, rb) in enumerate(zip(a, b)) if ra['character'] != rb['character']}


def load(path):
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r['set'] != 'strict':
            continue
        out.setdefault(int(r['step']), {})[int(r['idx'])] = r
    return out


CH = changed_idx()
print(f'改标签样本的 idx: {sorted(CH)}  (共 {len(CH)} 条) -> 用其余 {249 - len(CH)} 条对比\n')

e0 = load(E0)
fx = load(FX)

print('=== strict 断笔比 (frag_ratio), 只用 229 条未改标签样本 ===')
print(f'{"step":>7} | {"E0 frag":>8} {"E0 ink":>7} | {"inj3-fix":>8} {"ink":>7} | {"Δfrag":>8}')
print('-' * 60)
for step in sorted(fx):
    a = e0.get(step, {})
    b = fx.get(step, {})
    keys = [k for k in b if k in a and k not in CH]
    if len(keys) < 20:
        continue
    af = np.mean([float(a[k]['frag_ratio']) for k in keys])
    bf = np.mean([float(b[k]['frag_ratio']) for k in keys])
    ai = np.mean([float(a[k]['ink_ssim']) for k in keys])
    bi = np.mean([float(b[k]['ink_ssim']) for k in keys])
    d = bf - af
    print(f'{step:>7} | {af:8.3f} {ai:7.4f} | {bf:8.3f} {bi:7.4f} | {d:+8.3f}')

print()
print('=== 配对检验 (逐样本 Δfrag, n=229) ===')
for step in sorted(fx):
    a = e0.get(step, {})
    b = fx.get(step, {})
    keys = [k for k in b if k in a and k not in CH]
    if len(keys) < 20:
        continue
    d = np.array([float(b[k]['frag_ratio']) - float(a[k]['frag_ratio']) for k in keys])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f'  step{step}: mean Δ={d.mean():+.4f}  SE={se:.4f}  t={d.mean() / se if se else 0:+.2f}  '
          f'(负=inj3-fixed 断笔更少)')

print()
print('=== 参考: E0 自己的断笔比轨迹 (全部 249 条, 旧集) ===')
for step in sorted(e0):
    v = np.mean([float(r['frag_ratio']) for r in e0[step].values()])
    mark = '  ← 判别区间' if step > 40000 else ''
    print(f'  step{step:>7}: {v:.3f}{mark}')
