#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 10: 末步总表 + v25@150k 泄漏拆分 (供文档引用的最终数字)."""
import csv, os, re, statistics as st

print('==== 各 run 最后一次 in-mem-eval 原始行 ====')
for f in ('v25_stdskel', 'v24_frozenskel', 'v24_top10_style23', 'v23_splitnorm', 'v22_aug_skelnet_200k',
          'v21_skelnet_200k', 'v18_series/v18_train_20260925-053355'):
    p = f'logs/{f}.log'
    if not os.path.exists(p):
        print(f'  {f}: 无日志'); continue
    lines = [l for l in open(p, errors='ignore') if '[in-mem-eval]' in l and 'ssim=' in l]
    last = {}
    for l in lines:
        m = re.search(r'set=(\w+)', l)
        if m: last[m.group(1)] = re.sub(r'\x1b\[[0-9;]*m', '', l).strip()
    print(f'  --- {f}')
    for k, v in last.items():
        i = v.find('step=')
        print('    ' + v[i:i+300])

print('\n==== v25 @150k: strict 84 拆 训练内(62) / 训练外(22) ====')
tp = set(os.path.basename(r['image_path']) for r in csv.DictReader(open('assets/train_top10_style23.csv')))
rows = [r for r in csv.DictReader(open('assets/results/v25_stdskel/eval_stdskel_batch.csv'))
        if r['set'] == 'strict' and int(r['step']) == 150000]
inn, out = [], []
for r in rows:
    (inn if os.path.basename(r['img_id']) in tp else out).append(r)
NUM = ('ssim', 'ink_ssim', 'ink_iou', 'ink_recall', 'frag_ratio', 'tgt_spec', 'std_spec')
for nm, g in (('训练内(泄漏)', inn), ('训练外(真泛化)', out)):
    v = lambda k: st.mean([float(r[k]) for r in g if r[k] not in ('', None)])
    print(f'  {nm:12s} n={len(g):3d} ' + ' '.join(f'{k}={v(k):.4f}' for k in NUM))

print('\n==== 同口径对照: v24_frozenskel @90k 同拆分 ====')
rows = [r for r in csv.DictReader(open('assets/results/v24_frozenskel/eval_stdskel_batch.csv'))
        if r['set'] == 'strict' and int(r['step']) == 90000]
inn, out = [], []
for r in rows:
    (inn if os.path.basename(r['img_id']) in tp else out).append(r)
for nm, g in (('训练内(泄漏)', inn), ('训练外(真泛化)', out)):
    v = lambda k: st.mean([float(r[k]) for r in g if r[k] not in ('', None)])
    print(f'  {nm:12s} n={len(g):3d} ' + ' '.join(f'{k}={v(k):.4f}' for k in NUM))
