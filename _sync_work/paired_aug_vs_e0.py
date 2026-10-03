#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对检验: inj3-aug vs E0（229 条未改标签样本，逐样本 Δfrag / Δssim）。"""
import csv
import os

import numpy as np

OLD = 'assets/eval_v13_strict.csv'
NEW = 'assets/eval_v13_strict_fixed.csv'
E0 = 'assets/results/v17_s2_s2z_baseline/eval_backfill_frag_per_sample.csv'
AG = 'assets/results/v17_inj3_fixed_aug_100k/eval_backfill_frag_per_sample.csv'
E0B = 'assets/results/v17_s2_s2z_baseline/eval_stdskel_batch.csv'
AGB = 'assets/results/v17_inj3_fixed_aug_100k/eval_stdskel_batch.csv'


def ch():
    a = list(csv.DictReader(open(OLD, encoding='utf-8')))
    b = list(csv.DictReader(open(NEW, encoding='utf-8')))
    return {i for i, (ra, rb) in enumerate(zip(a, b)) if ra['character'] != rb['character']}


def load(path, key):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r.get('set') != 'strict':
            continue
        k = int(r['idx']) if 'idx' in r else int(r['img_id'].split('/')[-1])
        out.setdefault(int(r['step']), {})[k] = r
    return out


CH = ch()
for tag, pa, pb, col in [('frag_ratio', E0, AG, 'frag_ratio'),
                         ('ssim', E0B, AGB, 'ssim'),
                         ('ink_ssim', E0B, AGB, 'ink_ssim')]:
    A, B = load(pa, col), load(pb, col)
    print(f'=== {tag}（229 条，Δ = inj3-aug − E0；负 = aug 更少断笔 / 更低）===')
    for s in sorted(B):
        if s not in A:
            continue
        keys = [k for k in B[s] if k in A[s] and k not in CH]
        if len(keys) < 20:
            continue
        d = np.array([float(B[s][k][col]) - float(A[s][k][col]) for k in keys])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  step{s}: n={len(d)}  mean Δ={d.mean():+.4f}  SE={se:.4f}  '
              f't={d.mean()/se if se else 0:+.2f}')
    print()
