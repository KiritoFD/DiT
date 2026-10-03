#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解码 shards_std 的 g 条件 latent，实测真实笔宽 —— 确认编码的是 ~4px 原图。"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

os.chdir('/root/Workspace/xy/DiT')
from diffusers.models import AutoencoderKL  # noqa: E402

print('[1] 载 VAE', flush=True)
vae = AutoencoderKL.from_pretrained(
    'data/pretrained/pretrained_models/sd-vae-ft-ema',
    local_files_only=True).eval()
print('[2] 载 shards', flush=True)
m = {}
for f in sorted(glob.glob('data/50k/shards_std/shard_*.npz')):
    z = np.load(f)
    # ★ 必须先缓存: npz 是懒加载, 循环里 z['latents'][j] 会**每次解压整个数组**
    #   -> 51036 次解压, 慢到看起来像卡死（这个坑踩了两次）
    L, I = z['latents'], z['img_ids']
    for j, i in enumerate(I):
        m[int(i)] = L[j]
    z.close()
print(f'[3] {len(m)} 条', flush=True)

ids = [0, 1, 2, 100, 550, 9999, 44447]
print(f'{"id":>7} | {"解码墨迹px":>10} {"2*maxEDT":>9} {"2*meanEDT":>9} | {"原png 2*meanEDT":>14}')
for i in ids:
    if i not in m:
        continue
    lat = torch.from_numpy(m[i].astype(np.float32))[None] / 0.18215
    with torch.no_grad():
        dec = vae.decode(lat).sample[0]
    img = ((dec.clamp(-1, 1) + 1) / 2).mean(0).numpy()
    ink = img < 0.5
    p = f'data/50k/std/{i:06d}.png'
    ref = ''
    if os.path.exists(p):
        a = np.asarray(Image.open(p).convert('L'))
        ri = a < 127
        rd = distance_transform_edt(ri)
        ref = f'{2*float(rd.mean()):14.2f}'
    if ink.sum() < 10:
        print(f'{i:>7} | 空 {ref}')
        continue
    d = distance_transform_edt(ink)
    print(f'{i:>7} | {int(ink.sum()):>10} {2*float(d.max()):9.2f} '
          f'{2*float(d.mean()):9.2f} | {ref}', flush=True)
