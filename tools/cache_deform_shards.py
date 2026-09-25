#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把离线形变头的输出**预算成 shards**（冻结场景的最优做法）。

## 为什么
冻结的头 = 确定映射: g' = deform(g_std, 书家表[y])。
每个训练样本的书家是固定的 -> **g' 也是固定的** -> 没必要在训练时反复算。
直接预算成 `data/50k/shards_deform_v5/`，训练端当普通条件加载:
  · 零额外算力（不占 GPU、不加 step 时间）
  · 模型里不用挂形变头、不用挂中间监督 loss
  · 想换更好的头 -> 重跑本脚本即可

## 顺便定位一个缺口
扩散训练里 Deform loss = 0.27，而离线训练器收敛到 0.157。
本脚本用**两个 std 源**各算一遍 MSE(g', inst_skel)：
  · shards_std        <- v20 训练配置用的就是这个
  · shards_std_fixed  <- 离线训练器用的那个
哪个对得上 0.157，缺口就出在另一边。

用法: python tools/cache_deform_shards.py --ckpt assets/deform_skel_v5.pt \
        --out data/50k/shards_deform_v5
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.model.deform_skel import DeformSkel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='assets/deform_skel_v5.pt')
ap.add_argument('--out', default='data/50k/shards_deform_v5')
ap.add_argument('--std-dirs', default='data/50k/shards_std,data/50k/shards_std_fixed')
ap.add_argument('--gt-dir', default='data/50k/shards_aux_skel3')
ap.add_argument('--style-emb', default='assets/callig_emb_pretrained_50k.pt')
ap.add_argument('--batch', type=int, default=2048)
ap.add_argument('--width', type=int, default=96)
ap.add_argument('--max-off', type=float, default=6.0, dest='max_off')
ap.add_argument('--res-cap', type=float, default=2.0, dest='res_cap')
ap.add_argument('--shard-size', type=int, default=10048, dest='shard_size')
a = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_bank(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


# 书家标签
rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
C2I = {}
for r in rows:
    C2I.setdefault(str(r.get('calligrapher', '')), len(C2I))
lab = {}
for r in rows:
    try:
        i = int(os.path.basename(r.get('image_path', '')).split('.')[0])
    except Exception:
        continue
    lab[i] = C2I.get(str(r.get('calligrapher', '')), 0)

# 头 + 风格表
sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
head = DeformSkel(cond_dim=128, ch=4, grid=32, residual=1,
                  width=a.width, max_off=a.max_off, res_cap=a.res_cap)
head.load_state_dict(sd['deform'], strict=True)
head = head.to(DEV).eval()
tab = torch.load(a.style_emb, map_location='cpu', weights_only=False)['embedding'].float().to(DEV)
print(f'[1] 头已载入 {a.ckpt} | 参数 {sum(p.numel() for p in head.parameters()):,}', flush=True)

GT = load_bank(a.gt_dir)
print(f'[1] GT 骨架 {len(GT)} 条', flush=True)

os.makedirs(a.out, exist_ok=True)
for std_dir in a.std_dirs.split(','):
    std_dir = std_dir.strip()
    if not std_dir:
        continue
    A = load_bank(std_dir)
    ids = sorted(i for i in A if i in lab)
    print(f'\n[2] std 源 {std_dir}: {len(A)} 条, 可用 {len(ids)}', flush=True)

    G = torch.from_numpy(np.stack([A[i] for i in ids])).to(DEV)
    Y = torch.tensor([lab[i] for i in ids], device=DEV)
    out = np.empty((len(ids), 4, 32, 32), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(ids), a.batch):
            j = min(s + a.batch, len(ids))
            out[s:j] = head(G[s:j], tab[Y[s:j]]).half().cpu().numpy()
    print(f'    形变完成 {len(ids)} 条 {time.time() - t0:.0f}s', flush=True)

    common = [k for k, i in enumerate(ids) if i in GT]
    T = np.stack([GT[ids[k]] for k in common])
    base = float(np.mean((np.stack([A[ids[k]] for k in common]) - T) ** 2))
    mse = float(np.mean((out[common].astype(np.float32) - T) ** 2))
    print(f'    MSE(g_std, g_gt) = {base:.5f}  ->  MSE(g\', g_gt) = {mse:.5f}  '
          f'闭合率 {100 * (1 - mse / base):.1f}%', flush=True)

    if std_dir.endswith('shards_std'):
        n = len(ids)
        for k in range(0, n, a.shard_size):
            sl = slice(k, min(k + a.shard_size, n))
            p = os.path.join(a.out, f'shard_{k // a.shard_size:05d}.npz')
            np.savez(p, latents=out[sl],
                     img_ids=np.array([ids[t] for t in range(sl.start, sl.stop)],
                                      dtype=np.int64))
            print(f'    -> {p} {sl.stop - sl.start} 条', flush=True)
        print(f'[3] 已写 {a.out}（{n} 条）', flush=True)
