#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ink_census.py — 把"谁太瘦/太胖"量化：逐列统计墨量（暗像素占比）与对比度，基准是 GT。

结论用途：如果 GT 墨量明显高于 v68，则 v68 的 SSIM 落后 + SDE 的"加粗增益"都不是能力问题，
而是**系统性欠墨**；SDE 的 +0.02 就是"把欠的墨补回来"的度量假象。
"""

import os
import sys
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
BASE = f"{ROOT}/experiments/poster_4col"
COLS = [("v68_g0.0", 200000), ("v68_g0.5", 200000), ("v72_g0.0", 130000), ("v72_g0.5", 130000)]
N = 12


def ink_stats(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    dark = (a < 128).mean()                      # 墨量（暗像素占比）
    very_dark = (a < 64).mean()                  # 重墨
    return dark, very_dark, a.mean(), a.std()


def main():
    print(f"{'列':<10} {'墨量(<128)':>10} {'重墨(<64)':>10} {'均值':>7} {'std':>7}   相对 GT 墨量")
    gt = []
    for i in range(N):
        p = os.path.join(BASE, "v68_g0.0/eval_samples_ctrl/step0200000/poster", f"gt{i}.png")
        if os.path.exists(p):
            gt.append(ink_stats(p))
    if not gt:
        gt = [(0, 0, 0, 0)] * N
    g = np.array(gt)
    g_dark = g[:, 0].mean()
    print(f"{'GT':<10} {g_dark:>10.4f} {g[:,1].mean():>10.4f} {g[:,2].mean():>7.1f} {g[:,3].mean():>7.1f}    1.000")

    for label, step in COLS:
        vals = []
        for i in range(N):
            p = os.path.join(BASE, label, "eval_samples_ctrl", f"step{step:07d}", "poster", f"g{i}.png")
            if os.path.exists(p):
                vals.append(ink_stats(p))
        if not vals:
            print(f"{label:<10}  (缺图)")
            continue
        v = np.array(vals)
        d = v[:, 0].mean()
        print(f"{label:<10} {d:>10.4f} {v[:,1].mean():>10.4f} {v[:,2].mean():>7.1f} {v[:,3].mean():>7.1f}    {d/g_dark:>6.3f}")

    print()
    print("== 逐字(12)看 GT vs v68_g0.0 vs v72_g0.0 的墨量比 ==")
    print(f"{'#':>3} {'GT':>8} {'v68g0':>8} {'v68g0.5':>8} {'v72g0':>8} {'v72g0.5':>8}")
    for i in range(N):
        row = [i]
        for lab, step in [("GT", None)] + [(l, s) for l, s in COLS]:
            if lab == "GT":
                p = os.path.join(BASE, "v68_g0.0/eval_samples_ctrl/step0200000/poster", f"gt{i}.png")
            else:
                p = os.path.join(BASE, lab, "eval_samples_ctrl", f"step{step:07d}", "poster", f"g{i}.png")
            row.append(ink_stats(p)[0] if os.path.exists(p) else float("nan"))
        print(f"{row[0]:>3} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f} {row[4]:>8.4f} {row[5]:>8.4f}")


if __name__ == "__main__":
    main()
