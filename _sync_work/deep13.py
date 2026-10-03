#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 13: 泄漏复核的正确口径 —— 按 old_50k_id / src_image_path 比对同一张原图."""
import csv, os, collections

tr = list(csv.DictReader(open('assets/train_top10_style23.csv')))
print('train_top10 列名:', [c for c in tr[0].keys()])
tr_ids = set(r.get('old_50k_id', '') for r in tr if r.get('old_50k_id'))
tr_src = set(r.get('src_image_path', '') for r in tr if r.get('src_image_path'))
print(f'train_top10: unique old_50k_id={len(tr_ids)} unique src_image_path={len(tr_src)}')

for ev in ('assets/eval_top10_strict_subset84.csv', 'assets/eval_top10_seen_20.csv',
           'assets/eval_v13_strict_fixed.csv'):
    if not os.path.exists(ev):
        continue
    rr = list(csv.DictReader(open(ev)))
    id_hit = sum(1 for r in rr if r.get('old_50k_id') and r['old_50k_id'] in tr_ids)
    src_hit = sum(1 for r in rr if r.get('src_image_path') and r['src_image_path'] in tr_src)
    print(f'{os.path.basename(ev):34s} n={len(rr):3d}  '
          f'old_50k_id 命中={id_hit} ({100*id_hit/len(rr):.0f}%)  src 命中={src_hit} ({100*src_hit/len(rr):.0f}%)')

print('\n== 同一 old_50k_id 在 top10 训练里被扩了几张 (aug 变体) ==')
cnt = collections.Counter(r.get('old_50k_id') for r in tr if r.get('old_50k_id'))
print('  每原图训练变体数: max=', max(cnt.values()) if cnt else '-',
      ' med=', sorted(cnt.values())[len(cnt)//2] if cnt else '-')

print('\n== 50k 线自身复核: train_50k_v2_fixed vs eval_v13_strict (old_50k_id) ==')
tr50 = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv')))
i50 = set(r.get('old_50k_id') for r in tr50 if r.get('old_50k_id'))
rr = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv')))
hit = sum(1 for r in rr if r.get('old_50k_id') in i50)
print(f'  train_50k_v2_fixed unique old_50k_id={len(i50)}; eval_v13_strict 命中={hit}/{len(rr)}')
s50 = set(r.get('src_image_path') for r in tr50 if r.get('src_image_path'))
print(f'  src_image_path 命中={sum(1 for r in rr if r.get("src_image_path") in s50)}/{len(rr)}')

print('\n== 结论用: eval_v13_seen_fixed (seen 是否就是训练原图) ==')
rr2 = list(csv.DictReader(open('assets/eval_v13_seen_fixed.csv')))
print(f'  old_50k_id 命中 train_50k_v2_fixed = '
      f'{sum(1 for r in rr2 if r.get("old_50k_id") in i50)}/{len(rr2)}')
