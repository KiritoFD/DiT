#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复查 E1(style LN) vs E0 —— 用**新接进 eval 的那个函数**，匹配 step，带计数与检验。"""
import csv, glob, os, re, sys
import numpy as np
from PIL import Image
os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.in_mem_eval import _samechar_nn_eval, _SAMECHAR_BYCHAR

EV = 'assets/eval_v13_strict.csv'
ev = list(csv.DictReader(open(EV, encoding='utf-8')))[:249]
STEPS = [20000, 25000, 30000, 35000, 40000]
RUNS = [('E0', 'assets/results/v17_s2_s2z_baseline'), ('E1', 'assets/results/v17_s2z_ada1_ln')]


def load(rd, step):
    d = os.path.join(rd, 'eval_samples_ctrl', f'step{step:07d}', 'strict')
    if not os.path.isdir(d):
        return None
    P, G = [], []
    for i in range(249):
        gp, gtp = os.path.join(d, f'g{i}.png'), os.path.join(d, f'gt{i}.png')
        if not (os.path.exists(gp) and os.path.exists(gtp)):
            return None
        with Image.open(gp) as f: P.append(np.asarray(f.convert('RGB'), np.float32) / 255.0)
        with Image.open(gtp) as f: G.append(np.asarray(f.convert('RGB'), np.float32) / 255.0)
    return np.stack(P), np.stack(G)


print(f'{"run":>4} {"step":>7} | {"tgt_spec":>9} {"enrich":>7} {"hit":>8} {"base":>7} {"n_cal":>6} {"hits":>5}')
print('-' * 62)
res = {}
for nm, rd in RUNS:
    for st in STEPS:
        x = load(rd, st)
        if x is None:
            print(f'{nm:>4} {st:>7} | (无图)'); continue
        r = _samechar_nn_eval(x[0], x[1], ev, 'assets/train_50k_v2_fixed.csv')
        if r is None:
            print(f'{nm:>4} {st:>7} | (返回 None)'); continue
        res[(nm, st)] = r
        print(f'{nm:>4} {st:>7} | {r["tgt_spec"]:+9.4f} {r["cal_enrich"]:7.3f} '
              f'{r["cal_hit"]:7.1%} {r["cal_base"]:6.1%} {r["n_cal"]:>6} '
              f'{round(r["cal_hit"]*r["n_cal"]):>5}', flush=True)

print()
for k in ['tgt_spec', 'cal_enrich']:
    a = [res[('E1', s)][k] for s in STEPS if ('E1', s) in res]
    b = [res[('E0', s)][k] for s in STEPS if ('E0', s) in res]
    print(f'{k:>10}: E1 均值={np.mean(a):+.4f}  E0 均值={np.mean(b):+.4f}  '
          f'差={np.mean(a)-np.mean(b):+.4f}  逐step符号: '
          f'{["+" if x>y else "-" for x,y in zip(a,b)]}')

# 合并计数做二项检验
h1 = sum(round(res[('E1', s)]['cal_hit'] * res[('E1', s)]['n_cal']) for s in STEPS if ('E1', s) in res)
n1 = sum(res[('E1', s)]['n_cal'] for s in STEPS if ('E1', s) in res)
h0 = sum(round(res[('E0', s)]['cal_hit'] * res[('E0', s)]['n_cal']) for s in STEPS if ('E0', s) in res)
n0 = sum(res[('E0', s)]['n_cal'] for s in STEPS if ('E0', s) in res)
p1, p0 = h1 / n1, h0 / n0
print(f'\n合并 5 个 step 的命中计数: E1 {h1}/{n1} = {p1:.1%}   E0 {h0}/{n0} = {p0:.1%}')
se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
print(f'  差 {p1-p0:+.1%}  合并 SE {se:.1%}  -> {abs(p1-p0)/se:.2f} sigma')
print('  (注: 同一步两 run 用的评测样本相同, 严格应做配对检验; 这里给的是保守的独立近似)')
