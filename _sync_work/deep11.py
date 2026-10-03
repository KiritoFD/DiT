#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25/v24 泄漏拆分最终数字 (文档引用用)."""
import csv, os, statistics as st

tp = set(os.path.basename(r['image_path']) for r in
         csv.DictReader(open('assets/train_top10_style23.csv')))


def v(g, k):
    xs = []
    for r in g:
        s = r.get(k, '')
        if s not in ('', None) and s != 'nan':
            xs.append(float(s))
    return st.mean(xs) if xs else float('nan')


print('exp                step  组             n    ssim    ink_ssim ink_iou frag')
for exp, step in (('v25_stdskel', 150000), ('v25_stdskel', 90000),
                  ('v24_frozenskel', 90000), ('v24_top10_style23', 150000)):
    p = f'assets/results/{exp}/eval_stdskel_batch.csv'
    if not os.path.exists(p):
        continue
    rows = [r for r in csv.DictReader(open(p)) if r['set'] == 'strict' and int(r['step']) == step]
    if not rows:
        continue
    inn = [r for r in rows if os.path.basename(r['img_id']) in tp]
    out = [r for r in rows if os.path.basename(r['img_id']) not in tp]
    for nm, g in (('leak-in-train', inn), ('true-heldout', out)):
        if not g:
            continue
        print(f'{exp:18s} {step//1000:4d}k {nm:13s} {len(g):4d} '
              f'{v(g,"ssim"):.4f}  {v(g,"ink_ssim"):.4f}  {v(g,"ink_iou"):.4f} {v(g,"frag_ratio"):5.2f}')

print('\n== v25 seen/strict 末步 ==')
p = 'assets/results/v25_stdskel/eval_stdskel_batch.csv'
rows = list(csv.DictReader(open(p)))
for s in ('seen', 'strict'):
    g = [r for r in rows if r['set'] == s and int(r['step']) == 150000]
    if g:
        print(f'  {s:6s} n={len(g)} ssim={v(g,"ssim"):.4f} ink_ssim={v(g,"ink_ssim"):.4f} '
              f'ink_iou={v(g,"ink_iou"):.4f} frag={v(g,"frag_ratio"):.2f}')
