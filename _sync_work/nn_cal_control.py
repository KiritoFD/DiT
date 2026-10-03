#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照: strict 的每个字，训练集里到底有没有"同书家"的候选？

只有存在同书家候选的列, 才谈得上"最近邻是不是同书家"。
"""
import csv, os, sys, collections
import numpy as np
os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')

tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))[:249]
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)

n_has_cal = n_has_cal_script = n_has_any = n_nocand = 0
dist = []
for r in ev:
    ch, cal, sc = r.get('character'), r.get('calligrapher'), r.get('script')
    cs = by_char.get(str(ch), [])
    if not cs:
        n_nocand += 1; continue
    n_has_any += 1
    same_cal = [x for x in cs if x.get('calligrapher') == cal]
    same_both = [x for x in same_cal if x.get('script') == sc]
    if same_cal:
        n_has_cal += 1
    if same_both:
        n_has_cal_script += 1
    dist.append(len(cs))

N = len(ev)
print(f'strict {N} 列')
print(f'  训练集里**没有任何同字样本**: {n_nocand} ({n_nocand/N:.1%})')
print(f'  有同字候选:                 {n_has_any} ({n_has_any/N:.1%})')
print(f'  其中有**同书家**候选:        {n_has_cal} ({n_has_cal/N:.1%})')
print(f'  其中有**同书家+同书体**候选: {n_has_cal_script} ({n_has_cal_script/N:.1%})')
print(f'  每列的同字候选数: 中位={np.median(dist):.0f} 均值={np.mean(dist):.1f} max={max(dist)}')
print()
print('=> 若"同书家候选"只有 ~5%, 那"最近邻 4.4% 是同书家"就≈**在可选时全部命中**,')
print('   不能读成"没学到书家风格"。')
