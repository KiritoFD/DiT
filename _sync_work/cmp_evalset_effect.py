#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化「eval 标签修复」对 strict 指标的影响。

对比同一批 ckpt 在**旧评测集**与**修复评测集**上的分项，重点看那 20 条被修标签的样本。
"""
import csv
import sys

import numpy as np

A = 'assets/eval_v13_strict.csv'
B = 'assets/eval_v13_strict_fixed.csv'

a = list(csv.DictReader(open(A, encoding='utf-8')))
b = list(csv.DictReader(open(B, encoding='utf-8')))
fixed_ids = {ra['image_path'].split('/')[-1]
             for ra, rb in zip(a, b) if ra['character'] != rb['character']}
print(f'被修标签的样本数: {len(fixed_ids)}')


def stat(path, step, label):
    try:
        rows = [r for r in csv.DictReader(open(path, encoding='utf-8'))
                if r['set'] == 'strict' and r['step'] == step]
    except FileNotFoundError:
        print(f'{label}: 文件不存在 {path}')
        return
    if not rows:
        print(f'{label}: step {step} 无记录')
        return
    bad = [r for r in rows if r['img_id'].split('/')[-1] in fixed_ids]
    good = [r for r in rows if r['img_id'].split('/')[-1] not in fixed_ids]

    def m(g, k):
        return float(np.mean([float(r[k]) for r in g])) if g else float('nan')

    print(f'{label} step{step} n={len(rows)}')
    print(f'   修标签组 n={len(bad)}: ssim={m(bad, "ssim"):.4f} ink_ssim={m(bad, "ink_ssim"):.4f}')
    print(f'   其余     n={len(good)}: ssim={m(good, "ssim"):.4f} ink_ssim={m(good, "ink_ssim"):.4f}')
    print(f'   全体               : ssim={m(rows, "ssim"):.4f} ink_ssim={m(rows, "ink_ssim"):.4f}')
    return m(rows, 'ssim'), m(rows, 'ink_ssim')


print()
print('=== 旧评测集上的表现 ===')
e0_old = stat('assets/results/v17_s2_s2z_baseline/eval_stdskel_batch.csv', '100000', 'E0 旧集')
i3_old = stat('assets/results/v17_inj3_100k/eval_stdskel_batch.csv', '20000', 'inj3 旧集')
print()
print('=== 修复评测集上的表现 ===')
i3_new = stat('assets/results/v17_inj3_fixed_100k/eval_stdskel_batch.csv', '30000', 'inj3-fixed 修复集')
print()
print('★ 注意: 两套集子的数字不能直接相减(step 不同)。要精确量化需同一 ckpt 跑两套。')
print('   跑法: bash _sync_work/reeval_on_fixedset.sh INJ3_25K  (修复集)')
print('         再用默认 --sets 跑一遍同一 ckpt(旧集) 即可对齐。')
