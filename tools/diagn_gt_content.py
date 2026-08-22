#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 std_gt 骨架图 vs std_skel 图的真实内容形态, 检查拉回过程是否损坏。"""
import os, numpy as np
from PIL import Image

GTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fonttest", "std_gt", "kai")
STD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fonttest", "std_skel", "kai")

for f in sorted(os.listdir(GTD))[:3]:
    gp = os.path.join(GTD, f)
    g = np.asarray(Image.open(gp).convert("L"))
    print(f"\nGT {f}  shape={g.shape} dtype={g.dtype}")
    print(f"  值域 {g.min()}..{g.max()}, 唯一值数={len(np.unique(g))}")
    print(f"  <64像素数={int((g<64).sum())}, <128={int((g<128).sum())}, >128={int((g>128).sum())}, >200={int((g>200).sum())}")
    # 值的分布(采样)
    vals,bins=np.histogram(g.ravel(),bins=8,range=(0,256))
    print(f"  灰度直方图(8桶): {vals.tolist()}")

for f in sorted(os.listdir(STD))[:3]:
    sp = os.path.join(STD, f)
    s = np.asarray(Image.open(sp).convert("L"))
    print(f"\nSTD {f}  dtype={s.dtype} 值{int(s.min())}..{int(s.max())}")
    vals,bins=np.histogram(s.ravel(),bins=8,range=(0,256))
    print(f"  灰度直方图: {vals.tolist()}")
