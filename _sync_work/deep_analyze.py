#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep-dive analysis: latest eval rows per experiment + per-callig weaknesses."""
import csv, glob, json, os, re, collections

R = 'assets/results'

def show_cfg(name):
    p = f'src/train/configs/{name}.json'
    if not os.path.exists(p):
        print(f'-- {name}: no cfg'); return
    c = json.load(open(p))
    keys = ('experiment_name','max_steps','lr','_comment','skel_net','skel_encoder',
            'freeze_skel','w_style_rank','num_calligraphers','data_csv','skel_as_glyph_cond')
    print(f'-- {name}:')
    for k in keys:
        if k in c: print('   ', k, '=', str(c[k])[:120])

for n in ('v21_skelnet_200k','v22_aug_skelnet_200k','v23_splitnorm','v24_frozenskel','v25_stdskel'):
    show_cfg(n)

print('\n==== skelnet strict metrics csv (best row per exp/variant) ====')
f = f'{R}/eval_skelnet_strict_metrics.csv'
if os.path.exists(f):
    rows = list(csv.DictReader(open(f)))
    print('rows:', len(rows), 'cols:', list(rows[0].keys()) if rows else None)
    # aggregate mean ssim per variant column
    for col in ('ssim_std','ssim_true','ssim_xav','iou_std','iou_true','iou_xav'):
        try:
            v = [float(r[col]) for r in rows if r.get(col) not in (None,'','nan')]
            if v: print(f'  {col}: mean={sum(v)/len(v):.4f} n={len(v)}')
        except Exception as e:
            print(' ', col, 'err', e)

print('\n==== per-experiment latest in-mem eval ====')
import subprocess
pats = {
    'v18_style_rank': ['logs/v18_series/*.log'],
    'v19': ['logs/v19_series/*.log'],
    'v21': ['logs/v21_skelnet_200k.log'],
    'v22': ['logs/v22_aug_skelnet_200k.log'],
    'v23': ['logs/v23_splitnorm.log'],
    'v24fs': ['logs/v24_frozenskel.log'],
    'v24top10': ['logs/v24_top10_style23.log'],
    'v25': ['logs/v25*'],
}
for tag, gl in pats.items():
    files = []
    for g in gl: files += glob.glob(g)
    if not files:
        print(f'{tag}: no log'); continue
    latest = {}
    for fp in files:
        try: txt = open(fp, errors='ignore').read()
        except Exception: continue
        for m in re.finditer(r'\[in-mem-eval\] step=(\d+) set=(\w+) n=(\d+) ssim=([\d.]+).*?ink_ssim=([\d.]+) frag=([\d.]+).*?nn=([\d.]+) tgt_spec=([+\-\d.]+) cal_enrich=([\d.]+)x', txt):
            step, st = int(m.group(1)), m.group(2)
            latest[(st, step)] = m.groups()
    # print last 3 steps per set
    steps = sorted({k[1] for k in latest})
    for st in ('seen','strict'):
        for sp in steps[-3:]:
            g = latest.get((st, sp))
            if g:
                print(f'{tag:9s} {st:6s} step={g[0]:>7s} n={g[2]:>3s} ssim={g[3]} ink={g[4]} frag={g[5]} nn={g[6]} tgt_spec={g[7]} enrich={g[8]}x')

print('\n==== per-callig poster csv (strict) latest ====')
for d in sorted(glob.glob(f'{R}/*/posters/strict_train_samechar_nn.csv')):
    exp = d.split('/')[2]
    rows = list(csv.DictReader(open(d)))
    if not rows: continue
    cols = list(rows[0].keys())
    print(f'-- {exp}: {len(rows)} rows cols={cols[:8]}')
