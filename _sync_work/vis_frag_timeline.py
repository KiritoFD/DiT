#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""断笔比恶化的时间线: 固定一个样本, 把各 step 的生成图横排, 看碎裂从哪一步开始。

用法: python _sync_work/vis_frag_timeline.py <results_dir> --idx 42 [--set strict]
"""
import argparse
import os
import re

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('results_dir')
ap.add_argument('--idx', type=int, required=True)
ap.add_argument('--set', default='strict')
ap.add_argument('--per-row', type=int, default=7)
ap.add_argument('--out', default='_sync_work/frag_timeline.png')
a = ap.parse_args()

sd = os.path.join(a.results_dir, 'eval_samples_ctrl')
steps = sorted(int(re.search(r'step(\d+)', d).group(1))
               for d in os.listdir(sd) if d.startswith('step'))
steps = [s for s in steps if os.path.exists(
    os.path.join(sd, f'step{s:07d}', a.set, f'g{a.idx}.png'))]
print(f'{len(steps)} 个 step: {steps}')

W = 200
rows = (len(steps) + 1 + a.per_row - 1) // a.per_row
canvas = Image.new('RGB', (W * a.per_row, W * rows + 18), 'white')
dr = ImageDraw.Draw(canvas)
for k, s in enumerate(steps):
    r, c = divmod(k, a.per_row)
    p = os.path.join(sd, f'step{s:07d}', a.set, f'g{a.idx}.png')
    canvas.paste(Image.open(p).convert('RGB').resize((W, W)), (c * W, 18 + r * W))
    dr.text((c * W + 4, 18 + r * W + 3), f'{s // 1000}k', fill='red')
# 最后补一张 GT
r, c = divmod(len(steps), a.per_row)
gtp = os.path.join(sd, f'step{steps[0]:07d}', a.set, f'gt{a.idx}.png')
if os.path.exists(gtp):
    canvas.paste(Image.open(gtp).convert('RGB').resize((W, W)), (c * W, 18 + r * W))
    dr.text((c * W + 4, 18 + r * W + 3), 'GT', fill='blue')
dr.text((4, 3), f'idx{a.idx} ({a.set}) 各 step 生成图 —— 看碎裂起始', fill='black')
canvas.save(a.out)
print(f'-> {a.out}  ({canvas.size[0]}x{canvas.size[1]})')
