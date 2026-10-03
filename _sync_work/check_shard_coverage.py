#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验: 训练/评测 csv 需要的每个 id 是否都在骨架 shards 里。

shard 查表只认 img_id; 缺 id 时数据集**静默填零** -> 该样本 g=0, loss 照降、不报错。
这是"简繁/异体修复后 std_path 改指"最容易踩的坑, 必须显式查。
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.latent_dataset import extract_img_id  # noqa: E402


def shard_ids(d):
    ids = set()
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        ids.update(int(i) for i in z['img_ids'])
        z.close()
    return ids


def need(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
    return [extract_img_id(r) for r in rows], len(rows)


for shard_dir in ['data/50k/shards_std', 'data/50k/shards_std_fixed']:
    if not os.path.isdir(shard_dir):
        print(f'[skip] {shard_dir} 不存在')
        continue
    S = shard_ids(shard_dir)
    print(f'=== {shard_dir}: {len(S)} 个 id ===')
    for csvp in ['assets/train_50k_v2.csv', 'assets/train_50k_v2_fixed.csv',
                 'assets/eval_v13_strict.csv', 'assets/eval_v13_strict_fixed.csv',
                 'assets/eval_v13_seen.csv', 'assets/eval_v13_seen_fixed.csv']:
        if not os.path.exists(csvp):
            continue
        ids, n = need(csvp)
        miss = [i for i in ids if i not in S]
        flag = 'OK' if not miss else f'✗ 缺 {len(miss)} 个 -> 这些样本 g 会静默为 0'
        print(f'  {os.path.basename(csvp):34s} n={n:6d} 缺失={len(miss):5d}  {flag}')
        if miss[:5]:
            print(f'      样例: {miss[:5]}')
