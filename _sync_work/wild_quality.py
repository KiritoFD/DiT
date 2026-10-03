#!/usr/bin/env python
"""粗查 wild 各文件夹的图像质量信号: 尺寸一致性 / 灰度均值 / 墨量占比 / 近白占比"""
import os, sys, io
import numpy as np
from PIL import Image

WILD = "/root/Workspace/xy/HCSU/wild_extract"
CALS = sys.argv[1:] or sorted(os.listdir(WILD))
print(f"{'folder':<12}{'n':>5}{'size_uniq':>10}{'gray_mean':>10}{'ink<128':>9}{'near_white':>11}")
for cal in CALS:
    p = os.path.join(WILD, cal)
    fs = sorted(os.listdir(p))[:80]
    sizes, grays, inks, whites = set(), [], [], []
    for f in fs:
        im = Image.open(os.path.join(p, f)).convert("L")
        sizes.add(im.size)
        a = np.asarray(im, dtype=np.uint8)
        grays.append(a.mean())
        inks.append((a < 128).mean())
        whites.append((a > 245).mean())
    print(f"{cal:<12}{len(fs):>5}{len(sizes):>10}{np.mean(grays):>10.1f}"
          f"{np.mean(inks):>9.3f}{np.mean(whites):>11.3f}")