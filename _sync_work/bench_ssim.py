#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比: numpy 逐个 SSIM vs ssim_torch 批量 (CPU)。"""
import csv, os, sys, time
import numpy as np, torch
from PIL import Image
os.chdir('/root/Workspace/xy/DiT'); sys.path.insert(0, '.')
from src.eval.metrics import ssim as _ssim, ssim_torch as _ssim_t

tr = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
ev = list(csv.DictReader(open('assets/eval_v13_strict_fixed.csv', encoding='utf-8')))[:20]
by_char = {}
for r in tr:
    by_char.setdefault(str(r.get('character', '')), []).append(r)
gd = 'assets/results/v17_gq_100k/eval_samples_ctrl/step0065000/strict'

pairs = []
for i, r in enumerate(ev):
    gp = os.path.join(gd, f'g{i}.png')
    if not os.path.exists(gp):
        continue
    g = np.asarray(Image.open(gp).convert('RGB'), np.float32) / 255.0
    cs = by_char.get(str(r.get('character', '')), [])[:12]
    ts = []
    for x in cs:
        p = x.get('image_path', '')
        if os.path.exists(p):
            ts.append(np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0)
    if ts:
        pairs.append((g, ts))
n_ss = sum(len(t) for _, t in pairs)
print(f'{len(pairs)} 列, 共 {n_ss} 次 SSIM')

t0 = time.time()
r1 = [[_ssim(g, t) for t in ts] for g, ts in pairs]
t_np = time.time() - t0

t0 = time.time()
r2 = []
for g, ts in pairs:
    gb = torch.from_numpy(g).permute(2, 0, 1)[None].expand(len(ts), -1, -1, -1)
    tb = torch.from_numpy(np.stack(ts)).permute(0, 3, 1, 2)
    r2.append(_ssim_t(gb, tb).tolist())
t_t = time.time() - t0

d = max(abs(a - b) for x, y in zip(r1, r2) for a, b in zip(x, y))
print(f'numpy 逐个 : {t_np:6.2f}s')
print(f'batch torch: {t_t:6.2f}s   -> 加速 {t_np/max(t_t,1e-6):.1f}x')
print(f'两者数值最大差: {d:.2e}  (应≈0, 同口径)')
print(f'外推 249 列: {t_np/len(pairs)*249:.0f}s -> {t_t/len(pairs)*249:.0f}s')

if torch.cuda.is_available():
    try:
        t0 = time.time()
        r3 = []
        for g, ts in pairs:
            gb = torch.from_numpy(g).permute(2, 0, 1)[None].expand(len(ts), -1, -1, -1).cuda()
            tb = torch.from_numpy(np.stack(ts)).permute(0, 3, 1, 2).cuda()
            r3.append(_ssim_t(gb, tb).cpu().tolist())
        torch.cuda.synchronize()
        t_g = time.time() - t0
        d3 = max(abs(a - b) for x, y in zip(r2, r3) for a, b in zip(x, y))
        print(f'batch torch GPU: {t_g:6.2f}s -> 加速 {t_np/max(t_g,1e-6):.1f}x '
              f'| 外推 249 列 {t_g/len(pairs)*249:.1f}s | vs cpu-batch 差 {d3:.2e}')
    except Exception as e:
        print('GPU 测试失败:', type(e).__name__, str(e)[:80])
else:
    print('无 GPU')
