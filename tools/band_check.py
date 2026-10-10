#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""band_check.py — 验证 hf_losses 的 DoG 高通到底覆盖了哪个频段

问题: 训练内 HFr 用 highpass(k=7,sigma=1) 算 -> 显示 ~1.0
      离线普查用 FFT r>0.5 能量占比 -> 显示 0.788
      两者不同源。本脚本量化: DoG(k=7,sigma=1) 的**能量**里有多少落在 r>0.5 这个普查频段。
      若占比极小 -> 损失根本没在惩罚我们关心的那个频段(即 w-hf 白加)。
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def gauss_kernel(k, sigma):
    x = np.arange(k) - (k - 1) / 2.0
    w = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def lowpass(a, k, sigma):
    """可分离高斯低通（reflect padding）。"""
    ker = gauss_kernel(k, sigma)
    pad = k // 2
    b = np.pad(a, ((pad, pad), (0, 0)), mode="reflect")
    b = np.stack([np.convolve(b[:, c], ker, mode="valid") for c in range(a.shape[1])], axis=1)
    b = np.pad(b, ((0, 0), (pad, pad)), mode="reflect")
    return np.stack([np.convolve(b[r, :], ker, mode="valid") for r in range(a.shape[0])], axis=0)


def band_energy(a):
    """返回 (总能量, r>0.5 能量, r>0.35 能量, r>0.25 能量)"""
    b = a - a.mean()
    F = np.fft.fftshift(np.abs(np.fft.fft2(b)) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    tot = F.sum()
    return tot, F[r > 0.5].sum() / tot, F[r > 0.35].sum() / tot, F[r > 0.25].sum() / tot


def main():
    gts = [np.asarray(Image.open(f"eval/gt/{i}.png").convert("L"), dtype=np.float64) / 255.0 for i in range(30)]
    print(f"n={len(gts)} 张 GT\n")
    print(f"{'信号':<28}{'r>0.5占比':>11}{'r>0.35':>9}{'r>0.25':>9}{'相对全图能量':>13}")
    # 全图基准
    t = np.mean([band_energy(x)[0] for x in gts])
    q = np.mean([band_energy(x)[1] for x in gts])
    print(f"{'GT 原图(去均值)':<28}{q:>11.4f}"
          f"{np.mean([band_energy(x)[2] for x in gts]):>9.4f}{np.mean([band_energy(x)[3] for x in gts]):>9.4f}{1.0:>13.3f}")

    for k, sigma in ((7, 1.0), (5, 0.8), (3, 0.6), (3, 0.45)):
        hp = [x - lowpass(x, k, sigma) for x in gts]
        tot_hp = np.mean([band_energy(h)[0] for h in hp])
        print(f"{'DoG 高通 k=%d sigma=%.2f' % (k, sigma):<28}"
              f"{np.mean([band_energy(h)[1] for h in hp]):>11.4f}"
              f"{np.mean([band_energy(h)[2] for h in hp]):>9.4f}"
              f"{np.mean([band_energy(h)[3] for h in hp]):>9.4f}"
              f"{tot_hp/t:>13.4f}")

    # Laplacian
    lap = []
    for x in gts:
        L = np.zeros_like(x)
        L[1:-1, 1:-1] = (4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:])
        lap.append(L)
    tot_lap = np.mean([band_energy(h)[0] for h in lap])
    print(f"{'Laplacian(3x3)':<28}{np.mean([band_energy(h)[1] for h in lap]):>11.4f}"
          f"{np.mean([band_energy(h)[2] for h in lap]):>9.4f}{np.mean([band_energy(h)[3] for h in lap]):>9.4f}{tot_lap/t:>13.4f}")

    print("\n判读: 若 k=7,sigma=1 那一行的 r>0.5 占比远低于 GT 原图, 说明它主要在'中频',")
    print("      即 w-hf 的 L1 与原有 L1 高度共线 -> 对普查频段几乎没有额外梯度。")
    print("      Laplacian / 更小 k 的 DoG 才把能量压到 r>0.5。")


if __name__ == "__main__":
    main()
