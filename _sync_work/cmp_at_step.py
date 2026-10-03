#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各 run 在指定 step 的 strict 指标（**全 249 条均值**，不是单条）。"""
import csv
import os
import sys

import numpy as np

step = sys.argv[1] if len(sys.argv) > 1 else '5000'
RUNS = [
    ('E0', 'assets/results/v17_s2_s2z_baseline/eval_stdskel_batch.csv'),
    ('inj3(旧数据)', 'assets/results/v17_inj3_100k/eval_stdskel_batch.csv'),
    ('inj3-fixed', 'assets/results/v17_inj3_fixed_100k/eval_stdskel_batch.csv'),
    ('inj3-aug', 'assets/results/v17_inj3_fixed_aug_100k/eval_stdskel_batch.csv'),
    ('gq(本次)', 'assets/results/v17_gq_100k/eval_stdskel_batch.csv'),
]
print(f'=== step {step} 的 strict 指标（全 249 条均值）===')
print(f'{"run":>14} | {"n":>4} {"ssim":>8} {"ink_ssim":>9} {"frag":>8} {"hole":>7}')
for n, p in RUNS:
    if not os.path.exists(p):
        print(f'{n:>14} | 无文件')
        continue
    rows = [r for r in csv.DictReader(open(p, encoding='utf-8'))
            if r['step'] == step and r['set'] == 'strict']
    if not rows:
        print(f'{n:>14} | 无该 step')
        continue

    def m(k):
        v = [float(r[k]) for r in rows if r.get(k) not in (None, '', 'nan')]
        return float(np.mean(v)) if v else float('nan')
    print(f'{n:>14} | {len(rows):>4} {m("ssim"):8.4f} {m("ink_ssim"):9.4f} '
          f'{m("frag_ratio"):8.3f} {m("hole_pred"):7.4f}')
