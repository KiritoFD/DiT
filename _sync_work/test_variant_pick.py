#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实测: dataset 每步随机选骨架变体是否真的生效。

做法: 构造带 5 变体的 dataset, 反复取同一个 idx, 把返回的 skel_latent 与
_skel_latents_list[k][idx] 逐一比对, 看命中的 k 是否在 0..4 之间跳。
"""
import os
import sys

import numpy as np
import torch

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')

from src.train.cli import parse_args  # noqa: E402
from src.utils.latent_dataset import MCCDLatentDataset  # noqa: E402

a = parse_args(['--config', 'src/train/configs/v17_inj3_fixed_aug_100k.json'])
dirs = [d.strip() for d in str(a.skel_latent_shards_dirs).split(',') if d.strip()]
print(f'配置里的变体目录: {dirs}')

ds = MCCDLatentDataset(
    csv_file=a.data_csv, latent_shards_dir=a.latent_shards_dir,
    img_root=getattr(a, 'img_root', '') or '', image_size=256,
    preload=True, load_image=False,
    skel_latent_shards_dir=a.skel_latent_shards_dir,
    skel_latent_shards_dirs=dirs,
    callig_id_map=None,
    callig_script_map=(__import__('json').load(open(a.callig_script_map, encoding='utf-8'))
                       if getattr(a, 'callig_script_map', '') else None))

L = getattr(ds, '_skel_latents_list', None)
print(f'_skel_latents_list: {0 if L is None else len(L)} 个变体, 各 shape={None if L is None else L[0].shape}')

IDX = 12345
hit = []
for _ in range(40):
    b = ds[IDX]
    got = b['skel_latent'].numpy()
    found = -1
    for k, arr in enumerate(L):
        if np.allclose(arr[IDX], got, atol=1e-5):
            found = k
            break
    hit.append(found)
import collections
print(f'取 {IDX} 号样本 40 次的变体命中: {dict(collections.Counter(hit))}')
print('期望: 5 个变体都出现，且 -1（未命中）为 0')
