#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""功能验证: 用真实图直接调 _samechar_nn_eval, 看返回的 4 个量是否合理。"""
import csv, os, sys
import numpy as np
from PIL import Image
os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.in_mem_eval import _samechar_nn_eval

RD = 'assets/results/v17_gq_100k/eval_samples_ctrl/step0080000/strict'
ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))[:249]
P, G = [], []
for i in range(len(ev)):
    gp, gtp = os.path.join(RD, f'g{i}.png'), os.path.join(RD, f'gt{i}.png')
    if os.path.exists(gp) and os.path.exists(gtp):
        with Image.open(gp) as f: P.append(np.asarray(f.convert('RGB'), np.float32) / 255.0)
        with Image.open(gtp) as f: G.append(np.asarray(f.convert('RGB'), np.float32) / 255.0)
    else:
        P.append(np.zeros((256,256,3), np.float32)); G.append(np.zeros((256,256,3), np.float32))
ev = ev[:len(P)]
print(f'{len(P)} 列')
r = _samechar_nn_eval(np.stack(P), np.stack(G), ev, 'assets/train_50k_v2_fixed.csv')
print()
if r is None:
    print('✗ 返回 None'); sys.exit(1)
for k, v in r.items():
    print(f'  {k:12s} = {v}')
print()
print('合理性检查:')
print(f'  nn_ssim > nn_mean ?            {r["nn_ssim"] > r["nn_mean"]}  (max>=mean 必真)')
print(f'  tgt_spec > 0 ?                 {r["tgt_spec"] > 0}  (目标应比随机同字样本更近)')
print(f'  cal_enrich > 1 ?               {r["cal_enrich"] > 1}  (>1 = 书家信号存在)')
print(f'  nn_ssim 与 poster 里的一致 ?   poster 用最新 step 且走磁盘, 这里 step80000, 量级应吻合')
