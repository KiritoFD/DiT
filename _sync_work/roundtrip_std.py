#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测当前 g 条件（std png ~4px）的 VAE 往返保真度，并与 skeletonize 出的 1px/3px 对比。

往返 IoU = 原图墨迹 与 解码图墨迹 的 IoU。1px 应显著更低（用户方案的前提）。
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt

os.chdir('/root/Workspace/xy/DiT')
from diffusers.models import AutoencoderKL  # noqa: E402
from torchvision import transforms  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

print('[1] 加载 VAE ...', flush=True)
vae = AutoencoderKL.from_pretrained(
    'data/pretrained/pretrained_models/sd-vae-ft-ema',
    local_files_only=True).eval()
print('[2] VAE 就绪', flush=True)

tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])

IDS = [0, 1, 2, 100, 550, 1000, 9999, 44447]
pngs = [f'data/50k/std/{i:06d}.png' for i in IDS]
pngs = [p for p in pngs if os.path.exists(p)]
print(f'[3] 测 {len(pngs)} 张', flush=True)


def thicken(b, w):
    if w <= 1:
        return b
    d = distance_transform_edt(~b)
    return d <= (float(w) - 1.0) / 2.0


res = {}
for tag, tf_fn in [('orig(~4px)', None), ('skel 1px', 1), ('skel 3px', 3)]:
    ious = []
    for p in pngs:
        a = np.asarray(Image.open(p).convert('L'))
        ink = a < 127
        if tf_fn is not None:
            ink = thicken(skeletonize(ink), tf_fn)
        arr = np.where(ink, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, mode='L').convert('RGB')
        x = tf(im)[None]
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * 0.18215
            dec = vae.decode(lat).sample[0]
        rec = ((dec.clamp(-1, 1) + 1) / 2).mean(0).numpy() < 0.5
        inter = np.logical_and(ink, rec).sum()
        union = np.logical_or(ink, rec).sum()
        ious.append(float(inter) / max(float(union), 1.0))
    res[tag] = (float(np.mean(ious)), float(np.min(ious)))
    print(f'[4] {tag:12s} 往返 IoU mean={np.mean(ious):.4f} min={np.min(ious):.4f}', flush=True)

print()
print('=== 汇总 ===')
for k, (m, mn) in res.items():
    print(f'  {k:12s} mean={m:.4f}  min={mn:.4f}')
