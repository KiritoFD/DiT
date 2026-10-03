#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实测 std 骨架图的等效笔宽（EDT 法），判断当前 g 条件是几 px。"""
import csv
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
paths = [r['std_path'] for r in rows[:400]]
ws, inks = [], []
for p in paths:
    if not os.path.exists(p):
        continue
    a = np.asarray(Image.open(p).convert('L'))
    ink = a < 127
    if ink.sum() < 10:
        continue
    d = distance_transform_edt(ink)
    # 等效笔宽 = 2 * max(d) （细线情形）；对粗笔画取 2*mean(d[skel]) 更稳
    ws.append(2.0 * float(d.max()))
    inks.append(int(ink.sum()))
ws = np.array(ws)
inks = np.array(inks)
print(f'n={len(ws)}  std png 等效笔宽(2*max EDT): mean={ws.mean():.2f} p25={np.percentile(ws,25):.2f} '
      f'p50={np.percentile(ws,50):.2f} p75={np.percentile(ws,75):.2f} max={ws.max():.2f}')
print(f'  墨迹像素数: mean={inks.mean():.0f} p50={np.percentile(inks,50):.0f}')
print(f'  图像尺寸: {np.asarray(Image.open(paths[0]).convert("L")).shape}')
