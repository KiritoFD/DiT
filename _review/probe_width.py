#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_width.py — 目标骨架做多粗, latent 监督才立得住?

对同一批 GT 骨架 (strict84 PNG) 按 w = 3/7/11/15 px 各编码一次 latent, 每个宽度量:
  A. 排序: 墨迹投影 L1 与原始 MSE 下, gt < std < blank 是否成立。
  B. 可表示性: GT 骨架墨迹在 latent 里的**集中度** (能量最大的 5% 格子占比)。
     3px 的线折算到 32x32 只有 0.375 格 -> 墨被摊成低幅宽晕, 集中度低。
  C. 信号量: 与 std 的墨迹 L1 差 (要打败的就是它)。
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
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
try:
    from skimage.morphology import skeletonize
    HAVE_SKEL = True
except Exception:
    HAVE_SKEL = False

DEV = 'cuda'
SF = 0.18215
Z_BG = th.tensor([2.18129, 1.42018, -0.00979, -1.14073], device=DEV).view(1, 4, 1, 1)
D_INK = th.tensor([-3.16140, -4.00872, 1.12232, 2.44625], device=DEV).view(1, 4, 1, 1)
D_INK = D_INK / D_INK.norm()
DD = (D_INK * D_INK).sum().clamp_min(1e-9)

vae = AutoencoderKL.from_pretrained(
    'data/pretrained/pretrained_models/sd-vae-ft-ema').to(DEV).eval()
for p in vae.parameters():
    p.requires_grad_(False)

pngs = sorted(glob.glob('data/top10_style23/gt_skel_eval_strict84_png/*.png'))
print(f'GT 骨架 {len(pngs)} 张')


def skel_fallback(b):
    struct = generate_binary_structure(2, 2)
    sk = np.zeros_like(b)
    img = b.copy()
    while img.any():
        er = binary_erosion(img, structure=struct)
        sk |= img & ~er
        img = er
    return sk


def to_tensor(ink):
    a2 = np.where(ink, 0.0, 1.0).astype(np.float32)
    t = th.from_numpy(a2)[None, None].repeat(1, 3, 1, 1)
    t = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)
    return t * 2.0 - 1.0


@th.no_grad()
def encode(ts):
    out = []
    for i in range(0, len(ts), 32):
        x = th.cat(ts[i:i + 32]).to(DEV)
        out.append((vae.encode(x).latent_dist.mode() * SF).float().cpu())
    return th.cat(out, 0)


def ink(z):
    return ((z.to(DEV) - Z_BG) * D_INK).sum(1, keepdim=True) / DD


inks = []
for p in pngs:
    g = np.asarray(Image.open(p).convert('L'))
    b = g < 128
    if HAVE_SKEL and b.sum() > 0:
        b = skeletonize(b)
    inks.append(b)
print(f'skeletonize={HAVE_SKEL}  平均墨像素 {np.mean([b.sum() for b in inks]):.0f}')

std_pngs = [f'data/top10_style23/std/{os.path.basename(p)}' for p in pngs]
have_std = [p for p in std_pngs if os.path.exists(p)]
print(f'配套 std 骨架 {len(have_std)}/{len(std_pngs)}')

if have_std:
    sid = {os.path.basename(p): i for i, p in enumerate(std_pngs)}
    S_t = encode([to_tensor(np.asarray(Image.open(p).convert('L')) < 128)
                  for p in have_std])
    keep = [sid[os.path.basename(p)] for p in have_std]
    inks = [inks[i] for i in keep]
    print(f'配对后样本 {len(inks)}')
else:
    S_t = None

print()
print(f"{'w(px)':>6} {'lat格宽':>8} {'集中度':>8} {'MSE(s,b)':>9} {'MSE(gt,b)':>10} "
      f"{'L1(std,gt)':>11} {'L1(blank,gt)':>13} {'排序':>6}")
for w in (3, 7, 11, 15, 20, 25):
    it = max(0, (w - 1) // 2)
    T = encode([to_tensor(binary_dilation(b, iterations=it) if it > 0 else b)
                for b in inks])
    blank = Z_BG.expand_as(T).contiguous().cpu()
    i_t = ink(T).abs()
    flat = i_t.flatten(1)
    k = max(1, int(0.05 * flat.shape[1]))
    top = flat.topk(k, dim=1).values.sum(1)
    tot = flat.sum(1).clamp_min(1e-9)
    conc = float((top / tot).mean())
    mse_sb = float((S_t - blank).pow(2).mean()) if S_t is not None else float('nan')
    mse_tb = float((T - blank).pow(2).mean())
    if S_t is not None:
        l1_s = float((ink(S_t) - ink(T)).abs().mean())
    else:
        l1_s = float('nan')
    l1_b = float((ink(blank) - ink(T)).abs().mean())
    ok = 'OK' if (S_t is not None and l1_s < l1_b) else 'BAD'
    print(f'{w:>6} {w / 8.0:>8.2f} {conc:>8.3f} {mse_sb:>9.4f} {mse_tb:>10.4f} '
          f'{l1_s:>11.5f} {l1_b:>13.5f} {ok:>6}')
