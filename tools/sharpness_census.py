#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sharpness_census.py — 把"软/糊"量化：逐列量 边缘梯度能量、Laplacian 方差、高频能量占比，基准 GT。

用途：判断 v72 的欠缺是"结构/墨位"还是"高频细节(锐度)"，从而决定该上哪类推理端方法。
"""

import os
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
BASE = f"{ROOT}/experiments/poster_4col"
COLS = [("v68_g0.0", 200000), ("v68_g0.5", 200000), ("v72_g0.0", 130000), ("v72_g0.5", 130000)]
N = 12


def stats(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    grad = np.sqrt(gx ** 2 + gy ** 2).mean()                 # 边缘梯度能量
    lap = np.zeros_like(a)
    lap[1:-1, 1:-1] = (4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1]
                       - a[1:-1, :-2] - a[1:-1, 2:])
    lapvar = lap.var()                                        # 清晰度（Laplacian 方差）
    F = np.fft.fftshift(np.abs(np.fft.fft2(a - a.mean())) ** 2)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    hf = F[r > 0.5].sum() / max(F.sum(), 1e-9)                # 高频能量占比
    return grad, lapvar, hf


def collect(label, step):
    out = []
    for i in range(N):
        if label == "GT":
            p = os.path.join(BASE, "v68_g0.0/eval_samples_ctrl/step0200000/poster", f"gt{i}.png")
        else:
            p = os.path.join(BASE, label, "eval_samples_ctrl", f"step{step:07d}", "poster", f"g{i}.png")
        if os.path.exists(p):
            out.append(stats(p))
    return np.array(out)


def main():
    g = collect("GT", None)
    g_ref = g.mean(axis=0)
    print(f"{'列':<10} {'梯度能量':>9} {'相对GT':>7} {'Laplacian var':>14} {'相对GT':>7} {'高频占比':>9} {'相对GT':>7}")
    print(f"{'GT':<10} {g_ref[0]:>9.4f} {1.0:>7.2f} {g_ref[1]:>14.6f} {1.0:>7.2f} {g_ref[2]:>9.4f} {1.0:>7.2f}")
    for label, step in COLS:
        v = collect(label, step)
        if not len(v):
            print(f"{label:<10}  (缺图)"); continue
        m = v.mean(axis=0)
        print(f"{label:<10} {m[0]:>9.4f} {m[0]/g_ref[0]:>7.2f} {m[1]:>14.6f} {m[1]/g_ref[1]:>7.2f} "
              f"{m[2]:>9.4f} {m[2]/g_ref[2]:>7.2f}")


if __name__ == "__main__":
    main()
