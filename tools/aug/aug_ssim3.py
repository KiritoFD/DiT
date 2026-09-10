# -*- coding: utf-8 -*-
"""aug_ssim3.py — 变体 vs 原图的 SSIM (50 张子集): a / c / b1 / b2 / b3"""
import csv
import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssim_fn

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/5script/train_fame3_clean_v8.csv"
sys.stdout.reconfigure(encoding="utf-8")

LO, HI = 4.5, 9.0
MID = (LO + HI) / 2.0
STRENGTHS = {"b1": (0.6, 0.35), "b2": (1.2, 0.50), "b3": (2.0, 0.65)}

rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
rng = random.Random(7)
picks = rng.sample(range(len(rows)), 50)


def ink_domain(im):
    arr = 255.0 - np.asarray(im, dtype=np.float32)
    arr[arr < 8] = 0.0
    return arr


def stroke_width(arr):
    mask = arr > 40
    if mask.sum() < 20:
        return 0.0
    return float(np.median(ndi.distance_transform_edt(mask)[mask])) * 2.0


def make_a(im, rng):
    arr = np.clip(ink_domain(im) * rng.uniform(0.65, 1.45), 0, 255)
    return Image.fromarray((255.0 - arr).astype(np.uint8))


def make_c(im, rng):
    arr = ink_domain(im)
    ang = rng.uniform(-1.5, 1.5)
    dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
    arr = ndi.rotate(arr, ang, reshape=False, order=1, mode="constant", cval=0.0)
    arr = ndi.shift(arr, (dy, dx), order=1, mode="constant", cval=0.0)
    return Image.fromarray((255.0 - np.clip(arr, 0, 255)).astype(np.uint8))


def make_b(im, rng, kind):
    K, capf = STRENGTHS[kind]
    arr = ink_domain(im)
    w = stroke_width(arr)
    if w <= 0:
        return im
    target = float(np.clip(K * (MID - w), -capf * w, capf * w))
    passes = int(round(target / 2.0))
    if passes == 0 and abs(target) > 0.2:
        passes = int(np.sign(target))
    imc = Image.fromarray((255.0 - arr).astype(np.uint8))
    for _ in range(abs(passes)):
        imc = imc.filter(ImageFilter.MinFilter(3) if passes > 0 else ImageFilter.MaxFilter(3))
    arr = (ink_domain(imc) > 127) * 255.0
    return Image.fromarray((255.0 - arr).astype(np.uint8))


res = {k: [] for k in ("a", "c", "b1", "b2", "b3")}
for idx in picks:
    r = rows[idx]
    im = Image.open(os.path.join(ROOT, r["image_path"])).convert("L")
    o = np.asarray(im)
    rng2 = random.Random(10_000 + idx)
    for kind, fn in (("a", make_a), ("c", make_c)):
        v = np.asarray(fn(im, rng2))
        res[kind].append(ssim_fn(o, v, data_range=255))
    for kind in ("b1", "b2", "b3"):
        v = np.asarray(make_b(im, rng2, kind))
        res[kind].append(ssim_fn(o, v, data_range=255))

print(f"{'kind':>4} {'meanSSIM':>9} {'min':>7} {'max':>7}")
for k in ("a", "c", "b1", "b2", "b3"):
    v = np.array(res[k])
    print(f"{k:>4} {v.mean():9.4f} {v.min():7.4f} {v.max():7.4f}")
print("参考: 训练集同组合不同样本天花板 SSIM=0.745")
