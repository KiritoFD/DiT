#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 poster 最后样本(昌): sample4(生成) vs gt4(GT) vs remote_gt 的 canny/skel。
判断"推理给的不是昌"是生成质量差还是 GT 错位。
"""
import os, numpy as np
from PIL import Image

def load(p):
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)

def skel_frac(a):
    return float((a > 64).mean())

def cov(a, b):
    sa = a > 64; sb = b > 64
    inter = np.logical_and(sa, sb).sum()
    return inter / max(sb.sum(), 1)

HERE = os.path.dirname(os.path.abspath(__file__))
step = os.path.join(HERE, "remote_eval_samples", "step0005000")
# sample4 = 昌 生成; gt4 = 昌 GT原始
s4 = load(os.path.join(step, "sample4.png"))
g4 = load(os.path.join(step, "gt4.png"))
# remote_gt 的 101161(昌) canny/skel
rc = load(os.path.join(HERE, "remote_gt", "canny", "101161.png"))
rs = load(os.path.join(HERE, "remote_gt", "skel", "101161.png"))

for nm, im in [("sample4(生成)", s4), ("gt4(GT原图)", g4),
               ("remote_gt canny101161", rc), ("remote_gt skel101161", rs)]:
    if im is None: print(f"{nm}: missing"); continue
    print(f"{nm}: 墨像素占比={skel_frac(im):.3f} 值域[{im.min():.0f},{im.max():.0f}]")

if g4 is not None and s4 is not None:
    print(f"\nsample4(生成) 骨架覆盖 gt4(GT): {cov(s4,g4):.3f}")
    print(f"sample4(生成) 骨架覆盖 remote_gt skel: {cov(s4,rs):.3f}")
print("\n注意: 若 sample4 生成的结构和 gt4 一样是'昌'则覆盖应较高(>0.1 因生成质量); 若~0则生成的不是昌")
