#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对检验: inj3-fixed 与 E0 在**同一批 229 条未改标签样本**上的逐样本差。

两套集子在这 229 条上标签与骨架完全相同 -> 逐样本配对有效, 比"两个均值的差"紧得多。
E0 与 inj3-fixed 是**不同 seed 的两个 run**, 所以这里是"两 run 差"而非严格配对,
但同 step 同样本消除了样本难度差异, 仍是目前最紧的口径。
"""
import csv
import os

import numpy as np

E0 = 'assets/results/v17_s2_s2z_baseline/eval_stdskel_batch.csv'
FX = 'assets/results/v17_inj3_fixed_100k/eval_stdskel_batch.csv'
OLD = 'assets/eval_v13_strict.csv'
NEW = 'assets/eval_v13_strict_fixed.csv'


def fixed_ids():
    a = list(csv.DictReader(open(OLD, encoding='utf-8')))
    b = list(csv.DictReader(open(NEW, encoding='utf-8')))
    return {ra['image_path'].split('/')[-1]
            for ra, rb in zip(a, b) if ra['character'] != rb['character']}


def index(path, step):
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r['set'] == 'strict' and r['step'] == step:
            out[r['img_id'].split('/')[-1]] = r
    return out


FXID = fixed_ids()
print('配对口径: 229 条未改标签样本 (标签+骨架在两集子里完全相同)')
print(f'{"step":>7} | {"n":>4} | {"mean Δssim":>10} {"SE":>7} {"t":>6} | {"Δ>0 占比":>9}'
      f' | {"mean Δink":>10}')
print('-' * 74)
for step in sorted({r['step'] for r in csv.DictReader(open(FX, encoding='utf-8'))
                    if r['set'] == 'strict'}, key=int):
    a = index(E0, step)
    b = index(FX, step)
    keys = [k for k in b if k in a and k not in FXID]
    if len(keys) < 20:
        continue
    ds = np.array([float(b[k]['ssim']) - float(a[k]['ssim']) for k in keys])
    di = np.array([float(b[k]['ink_ssim']) - float(a[k]['ink_ssim']) for k in keys])
    se = ds.std(ddof=1) / np.sqrt(len(ds))
    t = ds.mean() / se if se > 0 else 0.0
    print(f'{step:>7} | {len(ds):>4} | {ds.mean():+10.4f} {se:7.4f} {t:6.2f} '
          f'| {(ds > 0).mean():9.1%} | {di.mean():+10.4f}')
print()
print('注: |t| > 2 才算在噪声外; E0 与 inj3-fixed 是不同 seed, 单步噪声本身就不小,')
print('    看趋势比看单点可靠。')
