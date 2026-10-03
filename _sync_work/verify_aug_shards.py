#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证增强 shards: id 覆盖 + 扰动是否真的生效 + 扰动幅度 vs 采样噪声。"""
import csv
import glob
import os
import sys

import numpy as np

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.utils.latent_dataset import extract_img_id  # noqa: E402


def load(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']      # ★ 必须先缓存（npz 懒加载）
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
need = [extract_img_id(r) for r in rows]
print(f'训练 csv: {len(need)} 条')

base = load('data/50k/shards_std_fixed')
print(f'shards_std_fixed: {len(base)} 条')

for v in ['el1', 'el2', 'aff', 'drp']:
    d = f'data/50k/shards_std_aug_{v}'
    if not os.path.isdir(d):
        print(f'{v:>4}: 还没建')
        continue
    m = load(d)
    miss = [i for i in need if i not in m]
    print(f'{v:>4}: {len(m)} 条 | 训练 csv 缺 {len(miss)} 条 | shard 数 '
          f'{len(glob.glob(d + "/shard_*.npz"))}')
    if miss[:5]:
        print(f'      缺样例 {miss[:5]}')
    common = [i for i in need[:3000] if i in m and i in base]
    if not common:
        continue
    dl = np.array([float(np.abs(m[i] - base[i]).max()) for i in common])
    el = np.array([float(np.abs(m[i] - base[i]).mean()) for i in common])
    print(f'      vs 原始: max|Δ| p50={np.percentile(dl,50):.4f} p90={np.percentile(dl,90):.4f} '
          f'| mean|Δ| p50={np.percentile(el,50):.4f}')
    n_same = int((dl < 0.05).sum())
    print(f'      几乎没变的条目: {n_same}/{len(common)}（应为 0 —— 否则扰动没生效）')
