#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""profile 同字最近邻: 时间花在 扫描csv / 载图 / SSIM 哪一段。"""
import csv, os, sys, time, collections
import numpy as np
from PIL import Image

os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.metrics import ssim as _ssim

t0 = time.time()
tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
t_csv = time.time() - t0
t0 = time.time()
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)
t_grp = time.time() - t0

ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))[:249]
chars = [str(r.get('character', '')) for r in ev]
cnt = collections.Counter(chars)
print(f'扫描训练 csv {len(tr)} 行: {t_csv:.2f}s | 按字分组: {t_grp:.2f}s')
print(f'strict 249 列 -> 唯一字 {len(cnt)} 个 | 重复最多的: {cnt.most_common(5)}')
print(f'  => 同字列数/唯一字 = {len(chars)/len(cnt):.2f}  (1.0 = 完全没有重复可省)')
cand_tot = sum(min(len(by_char.get(c, [])), 12) for c in chars)
print(f'  需要的候选总数: {cand_tot}  (去重后 {sum(min(len(by_char.get(c,[])),12) for c in set(chars))})')

# 载图 + SSIM 计时
gd = 'assets/results/v17_gq_100k/eval_samples_ctrl/step0065000/strict'
t_load = t_ssim = 0.0
n_ss = 0
t0 = time.time()
for i, c in enumerate(chars[:40]):
    gp = os.path.join(gd, f'g{i}.png')
    if not os.path.exists(gp):
        continue
    ta = time.time(); g = np.asarray(Image.open(gp).convert('RGB'), np.float32) / 255.0; t_load += time.time() - ta
    for x in by_char.get(c, [])[:12]:
        p = x.get('image_path', '')
        if not os.path.exists(p):
            continue
        ta = time.time(); t = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0; t_load += time.time() - ta
        ta = time.time(); _ssim(g, t); t_ssim += time.time() - ta; n_ss += 1
print(f'\n40 列的实测: 载图 {t_load:.2f}s | SSIM×{n_ss} {t_ssim:.2f}s '
      f'(单次 {t_ssim/max(n_ss,1)*1000:.1f}ms)')
print(f'  => 外推 249 列: 载图 {t_load/40*249:.0f}s | SSIM {t_ssim/40*249:.0f}s')
