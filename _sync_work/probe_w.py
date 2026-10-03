#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""各宽度 GT 骨架 PNG 的墨量 (纯 CPU)"""
import glob
import os
import sys

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
ids = sorted(os.path.basename(f)[:-4]
             for f in glob.glob("data/top10_style23/gt_skel_png/*.png"))[:300]
print(f"取样 {len(ids)} 个 id")
for d in ["gt_skel_png", "gt_skel_png_w1", "gt_skel_png_w3", "gt_skel_png_w7"]:
    fs = [f"data/top10_style23/{d}/{i}.png" for i in ids]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        print(f"  {d:>18}: 无文件")
        continue
    v = [(np.asarray(Image.open(f).convert("L")) < 128).mean() for f in fs]
    print(f"  {d:>18}: n={len(fs)}  ink={np.mean(v):.4f}  "
          f"(≈{np.mean(v)*256*256/np.mean([np.sqrt((np.asarray(Image.open(f).convert('L'))<128).sum()) for f in fs[:20]])**2:.2f}x 相对)")
