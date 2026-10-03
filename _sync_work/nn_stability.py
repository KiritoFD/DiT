#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同字最近邻随训练的稳定性分析。

问题: 每个 strict 样本"最像的训练集同字 GT"，随着训练推进会不会变？
  · 一直不变 -> 模型很早就锚定到某一张，之后不再移动（可能是记忆/吸引子）
  · 频繁变   -> 输出在训练集里游走（风格/结构还在漂）
并顺带看: 与最近邻的 SSIM 是否随训练上升（是否越来越像它）。
"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.eval.in_mem_eval import _train_nn_row  # noqa: E402

RD = sys.argv[1] if len(sys.argv) > 1 else 'assets/results/v17_gq_100k'
cfg = json.load(open(sorted(glob.glob(RD + '/*/resolved_config.json'))[-1], encoding='utf-8'))
TRAIN = cfg['data_csv']
EV = 'assets/eval_v13_strict_fixed.csv'
N = 249

steps = []
for d in sorted(glob.glob(os.path.join(RD, 'eval_samples_ctrl', 'step*'))):
    m = re.search(r'step(\d+)', os.path.basename(d))
    sub = os.path.join(d, 'strict')
    if m and os.path.isdir(sub):
        k = 0
        while os.path.exists(os.path.join(sub, f'g{k}.png')):
            k += 1
        if k:
            steps.append((int(m.group(1)), sub, k))
print(f'{len(steps)} 个 step: {[s[0] for s in steps]}')

ev = list(csv.DictReader(open(EV, encoding='utf-8')))[:N]
seq, sseq = {}, {}
for st, d, n in steps:
    r = _train_nn_row(TRAIN, EV, N, d)
    if r is None:
        print(f'  step{st}: 不可用'); continue
    paths, ssims, matched = r
    seq[st] = [os.path.basename(p) if p else None for p in paths]
    sseq[st] = ssims
    print(f'  step{st}: 平均 SSIM={np.nanmean(ssims):.4f}', flush=True)

STS = sorted(seq)
if len(STS) < 2:
    raise SystemExit('step 太少')

print('\n=== 1. 每个 strict 字的最近邻是否随训练变化 ===')
n_same = n_all_same = 0
nuniq = []
for i in range(N):
    s = [seq[st][i] for st in STS]
    u = len(set(s))
    nuniq.append(u)
    if u == 1:
        n_all_same += 1
nuniq = np.array(nuniq)
print(f'  全程最近邻**一次都没变**的列: {n_all_same}/{N} = {n_all_same/N:.1%}')
print(f'  每列的不同最近邻个数: mean={nuniq.mean():.2f} 中位={np.median(nuniq):.0f} '
      f'max={nuniq.max()} (共 {len(STS)} 个 step)')

print('\n=== 2. 相邻 step 之间的"换人率" ===')
for k in range(1, len(STS)):
    a, b = STS[k - 1], STS[k]
    ch = sum(1 for i in range(N) if seq[a][i] != seq[b][i])
    print(f'  {a:>7} -> {b:>7}: {ch:>3}/{N} = {ch/N:5.1%} 换了')

print('\n=== 3. 与"最终 step 的最近邻"的 SSIM 是否随训练上升 ===')
last = STS[-1]
for st in STS:
    # 把每列固定成最终 step 的最近邻，看 SSIM 怎么走
    same = [sseq[st][i] for i in range(N)
            if seq[st][i] == seq[last][i] and sseq[st][i] == sseq[st][i]]
    mean_all = np.nanmean(sseq[st])
    print(f'  step{st:>7}: 与该步最近邻 SSIM={mean_all:.4f}  '
          f'(其中 {len(same)} 列与最终最近邻相同, 其 SSIM 均值='
          f'{np.mean(same):.4f})' if same else
          f'  step{st:>7}: 与该步最近邻 SSIM={mean_all:.4f}')

print('\n=== 4. 最终最近邻的"书家是否与评测样本一致" ===')
lastm = _train_nn_row(TRAIN, EV, N, dict((s[0], s[1]) for s in steps)[last])
same_cal = sum(1 for i, m in enumerate(lastm[2])
               if m and m[1] == ev[i].get('calligrapher'))
print(f'  {same_cal}/{N} = {same_cal/N:.1%} 的列，最近邻是**同一书家**写的')
print('  (低 = 生成结果更像别的书家；高 = 确实学到了该书的笔法)')
