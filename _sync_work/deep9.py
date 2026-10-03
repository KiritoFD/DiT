#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnosis part 9: v26 配置 + top10 泄漏根因定位 + 受污染实验清单."""
import csv, json, os, collections

print('==== v26_gtskel 配置 ====')
for f in sorted(f for f in os.listdir('src/train/configs') if f.startswith('v26')):
    c = json.load(open('src/train/configs/' + f))
    for k in ('experiment_name','data_csv','max_steps','in_mem_eval_sets','deform_trainable',
              'w_deform_skel','skel_as_glyph_cond','glyph_inject_mode','num_calligraphers',
              'glyph_scale_init','cond_fusion','eval_cfg'):
        if k in c: print(f'  {k} = {str(c[k])[:100]}')
    print('  _comment:', c.get('_comment', '')[:420].replace('\n', ' '))

print('\n==== 泄漏根因: train_top10_style23.csv 的来源构成 ====')
rows = list(csv.DictReader(open('assets/train_top10_style23.csv')))
src = collections.Counter(r['source'] for r in rows)
print('  source 分布:', src.most_common(6))
paths = set(os.path.basename(r['image_path']) for r in rows)
print(f'  训练图数(basename 去重) = {len(paths)}')
for ev in ('assets/eval_top10_strict_subset84.csv', 'assets/eval_top10_seen_20.csv',
           'assets/eval_v13_strict_fixed.csv'):
    if not os.path.exists(ev): continue
    rr = list(csv.DictReader(open(ev)))
    hit = sum(1 for r in rr if os.path.basename(r['image_path']) in paths)
    print(f'  {os.path.basename(ev):34s} n={len(rr):3d} 命中 top10 训练集 = {hit} ({100*hit/len(rr):.0f}%)')

print('\n==== 同期: 50k 训练集 vs eval_v13_strict (应为 0) ====')
p50 = set(os.path.basename(r['image_path']) for r in csv.DictReader(open('assets/train_50k_v2_fixed.csv')))
rr = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv')))
print('  eval_v13_strict 命中 train_50k_v2_fixed =',
      sum(1 for r in rr if os.path.basename(r['image_path']) in p50), '/', len(rr))
p50b = set(os.path.basename(r['image_path']) for r in csv.DictReader(open('assets/train_50k_v2.csv')))
print('  eval_v13_strict 命中 train_50k_v2(增广前) =',
      sum(1 for r in rr if os.path.basename(r['image_path']) in p50b), '/', len(rr))
pa = set(os.path.basename(r['image_path']) for r in csv.DictReader(open('assets/train_50k_v2_fixed_augmented.csv'))) \
    if os.path.exists('assets/train_50k_v2_fixed_augmented.csv') else set()
if pa:
    print('  eval_v13_strict 命中 train_50k_v2_fixed_augmented (v22 用的) =',
          sum(1 for r in rr if os.path.basename(r['image_path']) in pa), '/', len(rr))

print('\n==== 受污染 run 清单 (用 top10 评测集) ====')
for f in sorted(os.listdir('src/train/configs')):
    if not f.endswith('.json'): continue
    c = json.load(open('src/train/configs/' + f))
    s = str(c.get('in_mem_eval_sets', '')) + str(c.get('data_csv', ''))
    if 'top10' in s:
        print(f'  {f:34s} data={c.get("data_csv")} eval={c.get("in_mem_eval_sets")}')
