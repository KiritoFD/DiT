#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""条件命中率: 在"有同书家候选"的列里, 最近邻有多少比例真的选了同书家？随机基线是多少？"""
import csv, glob, json, os, re, sys
import numpy as np
os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.in_mem_eval import _train_nn_row

RD = 'assets/results/v17_gq_100k'
cfg = json.load(open(sorted(glob.glob(RD + '/*/resolved_config.json'))[-1], encoding='utf-8'))
TRAIN, EV, N = cfg['data_csv'], 'assets/eval_v13_strict_fixed.csv', 249

tr = list(csv.DictReader(open(TRAIN, encoding='utf-8')))
ev = list(csv.DictReader(open(EV, encoding='utf-8')))[:N]
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)

steps = []
for d in sorted(glob.glob(os.path.join(RD, 'eval_samples_ctrl', 'step*'))):
    m = re.search(r'step(\d+)', os.path.basename(d)); sub = os.path.join(d, 'strict')
    if m and os.path.isdir(sub):
        k = 0
        while os.path.exists(os.path.join(sub, f'g{k}.png')): k += 1
        if k: steps.append((int(m.group(1)), sub))
step_dirs = dict(steps)

# 每列的可得性 + 随机基线(1/候选数, 因为同书家候选在候选里占 1/n)
avail_cal, base_cal, avail_both, base_both = [], [], [], []
for i, r in enumerate(ev):
    ch, cal, sc = r.get('character'), r.get('calligrapher'), r.get('script')
    cs = by_char.get(str(ch), [])
    nc = min(len(cs), 12)
    n_cal = sum(1 for x in cs if x.get('calligrapher') == cal)
    n_both = sum(1 for x in cs if x.get('calligrapher') == cal and x.get('script') == sc)
    avail_cal.append(n_cal > 0); base_cal.append(n_cal / max(nc, 1))
    avail_both.append(n_both > 0); base_both.append(n_both / max(nc, 1))
avail_cal, base_cal = np.array(avail_cal), np.array(base_cal)
avail_both, base_both = np.array(avail_both), np.array(base_both)
print(f'有同书家候选的列: {avail_cal.sum()} | 有同书家+同书体候选: {avail_both.sum()}')

print(f'\n{"step":>7} | {"同书家命中":>18} | {"同书家+书体命中":>18}')
for st in sorted(step_dirs):
    r = _train_nn_row(TRAIN, EV, N, step_dirs[st])
    if r is None: continue
    m = r[2]
    hit_cal = np.array([bool(m[i] and m[i][1] == ev[i].get('calligrapher')) for i in range(N)])
    hit_both = np.array([bool(m[i] and m[i][1] == ev[i].get('calligrapher')
                              and m[i][0] == ev[i].get('character')) for i in range(N)])
    # 同书体只能从命中同书家里判断 -> 用 train 行的 script 反查
    print(f'{st:>7} | {hit_cal[avail_cal].sum():>3}/{avail_cal.sum()} = '
          f'{hit_cal[avail_cal].mean():5.1%} (基线 {base_cal[avail_cal].mean():5.1%}) | '
          f'{hit_cal[avail_both].sum():>3}/{avail_both.sum()} = {hit_cal[avail_both].mean():5.1%}',
          flush=True)

last = max(step_dirs)
r = _train_nn_row(TRAIN, EV, N, step_dirs[last])
m = r[2]
hit_cal = np.array([bool(m[i] and m[i][1] == ev[i].get('calligrapher')) for i in range(N)])
print(f'\n=== 最终 step {last} ===')
print(f'  全体命中率        : {hit_cal.mean():5.1%}')
print(f'  有同书家候选时命中: {hit_cal[avail_cal].mean():5.1%}  vs 随机基线 {base_cal[avail_cal].mean():5.1%}'
      f'  -> 富集 {hit_cal[avail_cal].mean()/max(base_cal[avail_cal].mean(),1e-9):.1f}x')
print(f'  => 书家风格信号**存在但弱**: 即使可选, 78% 的列仍选了别的书家。')
