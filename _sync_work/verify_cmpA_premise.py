#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前提校验: 口径 A 用的 229 条样本, 其骨架 latent 在 shards_std 与 shards_std_fixed 里是否**完全相同**。

若不完全相同, "inj3-fixed vs E0 在 229 条上直接可比" 就不成立。
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.latent_dataset import extract_img_id  # noqa: E402

OLD = 'assets/eval_v13_strict.csv'
NEW = 'assets/eval_v13_strict_fixed.csv'


def load(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


a = list(csv.DictReader(open(OLD, encoding='utf-8')))
b = list(csv.DictReader(open(NEW, encoding='utf-8')))
changed = {ra['image_path'].split('/')[-1] for ra, rb in zip(a, b)
           if ra['character'] != rb['character']}

A = load('data/50k/shards_std')
B = load('data/50k/shards_std_fixed')

rows = [r for r in a if r['image_path'].split('/')[-1] not in changed]
print(f'未改标签样本: {len(rows)} 条')

diff, same, missing = [], 0, 0
for r in rows:
    i = extract_img_id(r)
    if i not in A or i not in B:
        missing += 1
        continue
    d = float(np.abs(A[i] - B[i]).max())
    if d > 0.2:
        diff.append((i, round(d, 3)))
    else:
        same += 1
print(f'  骨架相同(≤0.2): {same}')
print(f'  骨架不同(>0.2): {len(diff)}  {diff[:8]}')
print(f'  缺 id: {missing}')

# 改标签那 20 条: 应当**不同**
print()
c20 = [r for r in a if r['image_path'].split('/')[-1] in changed]
d20 = sum(1 for r in c20
          if extract_img_id(r) in A and extract_img_id(r) in B
          and float(np.abs(A[extract_img_id(r)] - B[extract_img_id(r)]).max()) > 0.2)
print(f'改标签样本: {len(c20)} 条, 其中骨架也变了 {d20} 条 (应当≈全部)')
print()
print('结论:', '前提成立 ✓' if not diff else '⚠ 前提不成立 —— 229 条里也有骨架变化, 口径 A 需修正')
