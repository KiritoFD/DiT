#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_autolevel.py — 选一种背景归一化，把 few-shot 主题的底色拉到 50k 的口径(底色 p90=255)。

50k 的图是**洗过底**的: 抽 401 张, 背景 p90 全部 = 255.0±0.0, 灰度均值≈204, 墨占比≈0.20。
wild 原图(含 50k 用过的同一批)背景 p90 只有 215-236。few-shot 若不洗底,
Diff/ssim 里就混进了"纸张灰度"这一项, 而且各主题灰度差很大(宋高宗 171 vs 伊秉绶-隶 246),
跨主题不可比。这里在几种候选里选一个既贴合 50k 又不动笔画形态的。
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageOps

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None


def stat(a):
    return (float(np.percentile(a, 90)), float(np.percentile(a, 95)),
            float(a.mean()), float((a < 128).mean()))


def show(label, vals):
    print(f"{label:<34} 底色p90={vals[0]:6.1f} 底色p95={vals[1]:6.1f} "
          f"灰均值={vals[2]:6.1f} 墨={vals[3]:.3f}")


def cand_none(a):
    return a


def cand_autocontrast(a):
    return np.asarray(ImageOps.autocontrast(Image.fromarray(a), cutoff=1), np.uint8)


def cand_p1_p95(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 95)
    return np.clip((a.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1e-3),
                   0, 255).astype(np.uint8)


def cand_p1_p90(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 90)
    return np.clip((a.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1e-3),
                   0, 255).astype(np.uint8)


CANDS = [("原样(只反相)", cand_none), ("autocontrast(cutoff=1)", cand_autocontrast),
         ("p1→p95 拉伸", cand_p1_p95), ("p1→p90 拉伸", cand_p1_p90)]

ref = [stat(np.asarray(Image.open(p).convert("L"), np.uint8))
       for p in sorted(glob.glob("data/50k/imgs/*.png"))[::80][:200]]
ref = np.mean(np.array(ref), 0)
show("★ 50k 训练图(参照)", ref)

for t in (sys.argv[1:] or ["沈周-行", "宋高宗-楷", "伊秉绶-隶"]):
    print(f"\n--- {t} ---")
    ps = sorted(glob.glob(f"data/50k/fs6_imgs_{t}/*.png"))[::10][:40]
    for name, fn in CANDS:
        acc = []
        for p in ps:
            a = np.asarray(Image.open(p).convert("L"), np.uint8)
            acc.append(stat(fn(a)))
        show(name, np.mean(np.array(acc), 0))
