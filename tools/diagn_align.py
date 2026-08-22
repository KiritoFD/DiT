#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 标准skel vs GT骨架对比为低的原因——是对齐/缩放问题还是真实字体差异。
对每个配对, 检查: 内容量、重心、外接框、以及平移后覆盖率上限。
"""
import os, numpy as np
from PIL import Image

STD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fonttest", "std_skel", "kai")
GT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fonttest", "std_gt", "kai")

def centroid(a):
    ys, xs = np.nonzero(a)
    if len(xs)==0: return None
    return ys.mean(), xs.mean(), ys.min(), ys.max(), xs.min(), xs.max()

def load(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    return a, (a>128)

for f in sorted(os.listdir(GT))[:6]:
    if not f.endswith(".png"): continue
    g = os.path.join(GT, f); s = os.path.join(STD, f)
    if not os.path.exists(s):
        continue
    ga, gb = load(g); sa, sb = load(s)
    gc = centroid(gb); sc = centroid(sb)
    # 内容量
    print(f"\n{f}")
    print(f"  标准: 墨像素={sb.sum()}, 重心行={sc[0] if sc else None:.0f} 列={sc[1] if sc else None:.0f}, 外接=Y[{sc[2] if sc else None:.0f},{sc[4] if sc else None:.0f}] X[{sc[3] if sc else None:.0f},{sc[5] if sc else None:.0f}]")
    print(f"  GT  : 墨像素={gb.sum()}, 重心行={gc[0] if gc else None:.0f} 列={gc[1] if gc else None:.0f}, 外接=Y[{gc[2] if gc else None:.0f},{gc[4] if gc else None:.0f}] X[{gc[3] if gc else None:.0f},{gc[5] if gc else None:.0f}]")
