#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三组对比 strict 断笔比（只用 229 条未改标签样本，绕开旧集/修复集口径断裂）。

E0                     : 旧数据 + 无增强          （旧评测集）
inj3-fixed             : 修复数据 + 无增强        （修复评测集）
inj3-fixed-aug         : 修复数据 + 全部条件增强   （修复评测集）
"""
import csv
import os

import numpy as np

OLD = 'assets/eval_v13_strict.csv'
NEW = 'assets/eval_v13_strict_fixed.csv'
RUNS = [
    ('E0', 'assets/results/v17_s2_s2z_baseline/eval_backfill_frag_per_sample.csv'),
    ('inj3-fixed', 'assets/results/v17_inj3_fixed_100k/eval_backfill_frag_per_sample.csv'),
    ('inj3-aug', 'assets/results/v17_inj3_fixed_aug_100k/eval_backfill_frag_per_sample.csv'),
    ('gq', 'assets/results/v17_gq_100k/eval_backfill_frag_per_sample.csv'),
]


def changed_idx():
    a = list(csv.DictReader(open(OLD, encoding='utf-8')))
    b = list(csv.DictReader(open(NEW, encoding='utf-8')))
    return {i for i, (ra, rb) in enumerate(zip(a, b)) if ra['character'] != rb['character']}


def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r['set'] == 'strict':
            out.setdefault(int(r['step']), {})[int(r['idx'])] = r
    return out


CH = changed_idx()
data = {n: load(p) for n, p in RUNS}
print('=== strict 断笔比 frag_ratio（229 条未改标签样本）===')
print(f'{"step":>7} | ' + ' | '.join(f'{n:>10}' for n, _ in RUNS) + ' |  Δ(aug-E0)  Δ(aug-fix)')
print('-' * 74)
steps = sorted({s for n, _ in RUNS for s in data[n]})
for s in steps:
    cells, aug, e0, fx = [], None, None, None
    for n, _ in RUNS:
        v = data[n].get(s, {})
        keys = [k for k in v if k not in CH]
        m = np.mean([float(v[k]['frag_ratio']) for k in keys]) if keys else None
        cells.append(f'{m:10.3f}' if m is not None else ' ' * 10)
        if n == 'E0':
            e0 = m
        elif n == 'inj3-fixed':
            fx = m
        elif n == 'inj3-aug':
            aug = m
    d1 = f'{aug - e0:+.3f}' if (aug is not None and e0 is not None) else '   n/a'
    d2 = f'{aug - fx:+.3f}' if (aug is not None and fx is not None) else '   n/a'
    print(f'{s:>7} | ' + ' | '.join(cells) + f' |  {d1:>9}  {d2:>9}')

print()
print('=== E0 的完整轨迹（参考：55k 后开始爆炸）===')
for s in sorted(data['E0']):
    v = data['E0'][s]
    keys = [k for k in v if k not in CH]
    m = np.mean([float(v[k]['frag_ratio']) for k in keys])
    print(f'  {s:>7}: {m:.3f}' + ('   ← 判别区间' if s > 40000 else ''))
