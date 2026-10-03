#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看 E0 断笔比爆炸长什么样: 挑 frag 涨幅最大的样本, 拼 GT | 40k | 100k 对比图。

用法: python _sync_work/vis_frag_boom.py <results_dir> [--step-a 40000] [--step-b 100000] [--top 4]
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('results_dir')
ap.add_argument('--set', default='strict')
ap.add_argument('--step-a', type=int, default=40000)
ap.add_argument('--step-b', type=int, default=100000)
ap.add_argument('--top', type=int, default=4)
ap.add_argument('--out', default='_sync_work/frag_boom.png')
a = ap.parse_args()

per = os.path.join(a.results_dir, 'eval_backfill_frag_per_sample.csv')
if not os.path.exists(per):
    raise SystemExit(f'✗ 先跑 backfill_frag.py 生成 {per}')

rows = [r for r in csv.DictReader(open(per, encoding='utf-8')) if r['set'] == a.set]
A = {int(r['idx']): r for r in rows if int(r['step']) == a.step_a}
B = {int(r['idx']): r for r in rows if int(r['step']) == a.step_b}
common = sorted(set(A) & set(B))
delta = sorted(common, key=lambda i: float(B[i]['frag_ratio']) - float(A[i]['frag_ratio']),
               reverse=True)
pick = delta[:a.top]
print(f'step{a.step_a} -> step{a.step_b}, 按 Δfrag 降序取 {a.top} 个:')
for i in pick:
    print(f'  idx={i:3d}  frag {float(A[i]["frag_ratio"]):6.3f} -> '
          f'{float(B[i]["frag_ratio"]):6.3f}   ink_ssim {float(A[i]["ink_ssim"]):.4f} -> '
          f'{float(B[i]["ink_ssim"]):.4f}')

sd = os.path.join(a.results_dir, 'eval_samples_ctrl')


def load(step, idx, kind):
    p = os.path.join(sd, f'step{step:07d}', a.set, f'{kind}{idx}.png')
    return Image.open(p).convert('RGB') if os.path.exists(p) else None


W = 256
canvas = Image.new('RGB', (W * 3, W * len(pick) + 22), 'white')
dr = ImageDraw.Draw(canvas)
for h, t in enumerate([f'GT ({a.set})', f'pred @{a.step_a}', f'pred @{a.step_b}']):
    dr.text((W * h + 6, 4), t, fill='black')
for row, i in enumerate(pick):
    y = 22 + row * W
    ims = [load(a.step_b, i, 'gt'), load(a.step_a, i, 'g'), load(a.step_b, i, 'g')]
    for col, im in enumerate(ims):
        if im is None:
            continue
        canvas.paste(im.resize((W, W)), (col * W, y))
    dr.text((6, y + 4),
            f'idx{i}: frag {float(A[i]["frag_ratio"]):.2f}->{float(B[i]["frag_ratio"]):.2f}',
            fill='red')
canvas.save(a.out)
print(f'-> {a.out}  ({canvas.size[0]}x{canvas.size[1]})')
