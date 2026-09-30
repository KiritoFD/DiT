#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_gt_recipe.py — 反推验证 GT 骨架 latent 的构建配方。

如果我的配方 (GT 骨架 PNG -> 骨架化 -> 膨胀 w -> VAE编码) 能重建出
shards_aux_skel3 里同 id 的 latent, 就可以放心用同一配方铺 7px 目标。
对不上 -> 说明目标空间不是这么来的, 必须先查清。
"""
import glob
import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '/root/Workspace/xy/DiT')
from diffusers.models import AutoencoderKL
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

DEV = 'cuda'
SF = 0.18215

# 读 aux_skel3 全库为 id -> latent
tab = {}
for f in sorted(glob.glob('data/top10_style23/shards_aux_skel3/shard_*.npz')):
    z = np.load(f)
    for j, iid in enumerate(z['img_ids']):
        tab[int(iid)] = z['latents'][j]
print(f'aux_skel3: {len(tab)} 条')

pngs = sorted(glob.glob('data/top10_style23/gt_skel_eval_strict84_png/*.png'))
ids = []
for p in pngs:
    try:
        ids.append(int(os.path.basename(p).split('.')[0]))
    except ValueError:
        pass
ids = [i for i in ids if i in tab]
print(f'strict84 PNG 命中 aux_skel3: {len(ids)}/{len(pngs)}')
if not ids:
    print('!! 没有交集 -> 两套 id 体系不同, 换用 img_id 列或直接比对统计量')
    sys.exit(0)

vae = AutoencoderKL.from_pretrained(
    'data/pretrained/pretrained_models/sd-vae-ft-ema').to(DEV).eval()
for p in vae.parameters():
    p.requires_grad_(False)


def to_tensor(ink):
    a2 = np.where(ink, 0.0, 1.0).astype(np.float32)
    t = th.from_numpy(a2)[None, None].repeat(1, 3, 1, 1)
    t = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)
    return t * 2.0 - 1.0


@th.no_grad()
def encode(ts):
    xs = th.cat(ts).to(DEV)
    return (vae.encode(xs).latent_dist.mode() * SF).float().cpu()


for w in (1, 3, 5):
    it = max(0, (w - 1) // 2)
    ts, ref = [], []
    for i, p in zip(ids, [q for q in pngs if int(os.path.basename(q).split('.')[0]) in ids]):
        g = np.asarray(Image.open(p).convert('L'))
        b = g < 128
        if b.sum() > 0:
            b = skeletonize(b)
        if it > 0:
            b = binary_dilation(b, iterations=it)
        ts.append(to_tensor(b))
        ref.append(torch.from_numpy(tab[i]).float())
    Z = encode(ts)
    R = th.stack(ref)
    mse = float((Z - R).pow(2).mean())
    cos = float(F.cosine_similarity(Z.flatten(1), R.flatten(1), dim=1).mean())
    print(f'  w={w}: MSE {mse:.5f}  cos {cos:.4f}  '
          f'(Z std {float(Z.std()):.4f} vs ref std {float(R.std()):.4f})')
