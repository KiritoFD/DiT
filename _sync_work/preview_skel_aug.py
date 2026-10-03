#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""std 骨架条件扰动：实测笔宽 + 生成各扰动变体的预览图。

设计原则（用户给的）：
  「在数据构建期保证信号不丢失，在训练前向期破坏其精确性」
  - 构建期：3px（skeletonize 后 binary_dilation 1 次）—— 1px 在 VAE 里会断裂
  - 前向期：局部 mask(patch drop) / 全局 mask(dropout) / 对称加噪(noise)

用法: python _sync_work/preview_skel_aug.py [--n-char 3]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (binary_dilation, distance_transform_edt,
                           gaussian_filter, map_coordinates, rotate)

ap = argparse.ArgumentParser()
ap.add_argument('--n-char', type=int, default=3)
ap.add_argument('--out', default='_sync_work/skel_aug_preview.png')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def skel_impl():
    try:
        from skimage.morphology import skeletonize
        return lambda b: skeletonize(b)
    except Exception:
        from scipy.ndimage import binary_erosion
        def _sk(b):
            img = b.copy()
            sk = np.zeros_like(img)
            st = np.ones((3, 3), bool)
            while img.any():
                er = binary_erosion(img, st)
                sk |= img & ~er
                img = er
            return sk
        return _sk


SKEL = skel_impl()


def to_ink(png):
    """白底黑线 -> bool(True=线)"""
    return np.asarray(Image.open(png).convert('L')) < 127


def to_png(arr_bool):
    return Image.fromarray(np.where(arr_bool, 0, 255).astype(np.uint8), mode='L')


def thicken(b, width):
    if width <= 1:
        return b
    d = distance_transform_edt(~b)
    return d <= (float(width) - 1.0) / 2.0


def elastic(b, alpha, sigma, rng):
    """Simard 式弹性形变（与 tools/aug6.py 同实现）。b: bool"""
    f = b.astype(np.float32)
    h, w = f.shape
    dx = gaussian_filter((rng.random((h, w)) - 0.5) * 2, sigma) * alpha
    dy = gaussian_filter((rng.random((h, w)) - 0.5) * 2, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    idx = [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
    out = map_coordinates(f, idx, order=1, mode='reflect')
    return out > 0.5


def affine(b, deg, scale, shear_deg, rng):
    out = b.astype(np.float32)
    out = rotate(out, deg, reshape=False, order=1, mode='constant', cval=0.0)
    sh = np.tan(np.deg2rad(shear_deg))
    h, w = out.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx, cy = w / 2, h / 2
    x2 = cx + (xx - cx) - sh * (yy - cy)
    out = map_coordinates(out, [yy, np.clip(x2, 0, w - 1)], order=1, mode='reflect')
    # 缩放（绕中心）
    yy2 = cy + (yy - cy) / scale
    xx2 = cx + (xx - cx) / scale
    out = map_coordinates(out, [np.clip(yy2, 0, h - 1), np.clip(xx2, 0, w - 1)],
                          order=1, mode='reflect')
    return out > 0.5


def patch_drop(b, p, rng, patch=8):
    """模拟 latent 级 patch drop: 1 latent patch = 8x8 像素，逐块丢。"""
    h, w = b.shape
    keep = rng.random((h // patch, w // patch)) > p
    m = np.kron(keep, np.ones((patch, patch), bool))
    return b & m


def region_drop(b, n_rect, size, rng):
    """整块 Cutout（模拟"丢一个笔画/局部区域"）。"""
    out = b.copy()
    h, w = b.shape
    for _ in range(n_rect):
        y = rng.integers(0, h - size)
        x = rng.integers(0, w - size)
        out[y:y + size, x:x + size] = False
    return out


def add_noise_px(b, sigma, rng):
    """像素级加噪（训练期实际是 latent 加噪，这里只是预览量级）。"""
    f = b.astype(np.float32) + rng.normal(0, sigma, b.shape).astype(np.float32)
    return f > 0.5


def width_stats(b):
    d = distance_transform_edt(b)
    if b.sum() == 0:
        return 0.0, 0.0
    sk = SKEL(b)
    return 2.0 * float(d.max()), (2.0 * float(d[sk].mean()) if sk.sum() else 0.0)


rows_csv = [l.split(',') for l in open('assets/train_50k_v2_fixed.csv',
                                       encoding='utf-8').read().split('\n')[1:] if l.strip()]
paths, chars = [], []
seen = set()
for r in rows_csv:
    if len(r) < 4:
        continue
    ch = r[3]
    sp = r[9] if len(r) > 9 else ''
    if ch in seen or not sp:
        continue
    seen.add(ch)
    paths.append(sp)
    chars.append(ch)
    if len(paths) >= a.n_char:
        break

rng = np.random.default_rng(a.seed)
print(f'取 {len(paths)} 个样本: {list(zip(chars, paths))}')

VARIANTS = [
    ('orig(现状)', lambda b: b),
    ('skel 1px', lambda b: SKEL(b)),
    ('skel 3px', lambda b: thicken(SKEL(b), 3)),
    ('skel 5px', lambda b: thicken(SKEL(b), 5)),
    ('elastic a8 s3', lambda b: elastic(b, 8, 3, rng)),
    ('elastic a16 s4', lambda b: elastic(b, 16, 4, rng)),
    ('affine 5d/s.95', lambda b: affine(b, 5, 0.95, 3, rng)),
    ('patchdrop .2', lambda b: patch_drop(b, 0.2, rng)),
    ('regiondrop 48', lambda b: region_drop(b, 1, 48, rng)),
]

print()
print(f'{"variant":>16} | ' + ' '.join(f'{c:>14}' for c in chars))
print('-' * (18 + 15 * len(chars)))
imgs = []
for name, fn in VARIANTS:
    row = []
    stats = []
    for p in paths:
        ink = to_ink(p)
        out = fn(ink)
        row.append(out)
        wmax, wmean = width_stats(out)
        stats.append(f'{wmax:5.1f}/{wmean:5.2f}')
    imgs.append((name, row))
    print(f'{name:>16} | ' + ' '.join(f'{s:>14}' for s in stats))

W = 200
canvas = Image.new('RGB', (W * len(paths), W * len(imgs) + 16), 'white')
dr = ImageDraw.Draw(canvas)
for r, (name, row) in enumerate(imgs):
    for c, im in enumerate(row):
        canvas.paste(to_png(im).convert('RGB').resize((W, W)), (c * W, 16 + r * W))
    dr.text((4, 16 + r * W + 3), name, fill='red')
for c, ch in enumerate(chars):
    dr.text((c * W + 4, 3), ch, fill='blue')
canvas.save(a.out)
print(f'\n-> {a.out}  ({canvas.size[0]}x{canvas.size[1]})')
print('列头: 等效笔宽 = 2*max(EDT) / 2*mean(EDT@骨架)')
