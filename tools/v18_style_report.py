#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v18(style-rank) 的风格指标，与同 step 基线对照。

用新接进 eval 的同一个函数 _samechar_nn_eval，只在 229 条共有列上算，保证可比。
"""
import csv
import os
import re
import sys

import numpy as np
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.eval.in_mem_eval import _samechar_nn_eval  # noqa: E402

_a = list(csv.DictReader(open('assets/eval_v13_strict.csv', encoding='utf-8')))
_b = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))
CHANGED = {i for i, (x, y) in enumerate(zip(_a, _b)) if x['character'] != y['character']}

CASES = [
    ('E0 @100k', 'assets/results/v17_s2_s2z_baseline', 100000, 'assets/eval_v13_strict.csv'),
    ('inj3-aug @100k', 'assets/results/v17_inj3_fixed_aug_100k', 100000, 'assets/eval_v13_strict_fixed.csv'),
    ('gq @85k', 'assets/results/v17_gq_100k', 85000, 'assets/eval_v13_strict_fixed.csv'),
    ('**v18 @100k**', 'assets/results/v18_style_rank_200k', 100000, 'assets/eval_v13_strict_fixed.csv'),
]


def load(rd, step, evp):
    d = os.path.join(rd, 'eval_samples_ctrl', 'step%07d' % step, 'strict')
    if not os.path.isdir(d):
        return None
    ev = list(csv.DictReader(open(evp, encoding='utf-8')))
    P, G, rows = [], [], []
    for i in range(min(249, len(ev))):
        gp, gtp = os.path.join(d, 'g%d.png' % i), os.path.join(d, 'gt%d.png' % i)
        if not (os.path.exists(gp) and os.path.exists(gtp)):
            continue
        with Image.open(gp) as f:
            P.append(np.asarray(f.convert('RGB'), np.float32) / 255.0)
        with Image.open(gtp) as f:
            G.append(np.asarray(f.convert('RGB'), np.float32) / 255.0)
        rows.append(ev[i])
    if not P:
        return None
    return np.stack(P), np.stack(G), rows


print('%-16s | %6s %4s | %9s %9s %9s | %7s %7s %6s' %
      ('run', 'step', 'n', 'own_gt', 'nn_max', 'nn_mean', 'tgt_spec', 'cal_enr', 'n_cal'))
print('-' * 92)
res = {}
for tag, rd, st, evp in CASES:
    x = load(rd, st, evp)
    if x is None:
        print('%-16s | (缺逐样本图)' % tag)
        continue
    P, G, rows = x
    keep = [i for i in range(len(rows)) if i not in CHANGED]
    r = _samechar_nn_eval(P[keep], G[keep], [rows[i] for i in keep],
                          'assets/train_50k_v2_fixed.csv')
    if r is None:
        print('%-16s | (返回 None)' % tag)
        continue
    own = float(np.mean([_ssim_single(P[i], G[i]) for i in keep])) if False else float('nan')
    res[tag] = r
    print('%-16s | %6d %4d | %9s %9.4f %9.4f | %+9.4f %7.2fx %6d' %
          (tag, st, r['n_ok'], '-', r['nn_ssim'], r['nn_mean'],
           r['tgt_spec'], r['cal_enrich'], r['n_cal']), flush=True)

print()
if '**v18 @100k**' in res and 'inj3-aug @100k' in res:
    a, b = res['**v18 @100k**'], res['inj3-aug @100k']
    print('v18 vs inj3-aug（同 100k, 唯一差异 = 加了 style-rank loss）:')
    print('  tgt_spec  %+.4f -> %+.4f  (%+.4f)' % (b['tgt_spec'], a['tgt_spec'],
                                                   a['tgt_spec'] - b['tgt_spec']))
    print('  cal_enrich %.2fx -> %.2fx  (%+.2fx)' % (b['cal_enrich'], a['cal_enrich'],
                                                     a['cal_enrich'] - b['cal_enrich']))
if '**v18 @100k**' in res and 'E0 @100k' in res:
    a, b = res['**v18 @100k**'], res['E0 @100k']
    print('v18 vs E0（同 100k）:')
    print('  tgt_spec  %+.4f -> %+.4f  (%+.4f)' % (b['tgt_spec'], a['tgt_spec'],
                                                   a['tgt_spec'] - b['tgt_spec']))
    print('  cal_enrich %.2fx -> %.2fx  (%+.2fx)' % (b['cal_enrich'], a['cal_enrich'],
                                                     a['cal_enrich'] - b['cal_enrich']))
