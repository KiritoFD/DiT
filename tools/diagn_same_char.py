#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""深度诊断: 对几个高频常用字(一/二/三/十/人/大/永)验证标准和GT是否同字拓扑。
用 行/列投影 分析: 同字拓扑的笔画分布应大致对应(横画=行集中/竖画=列集中)。"""
import os, numpy as np
from PIL import Image

HERE=os.path.dirname(os.path.abspath(__file__))
STD=os.path.join(HERE,"_fonttest","std_skel")
GTD=os.path.join(HERE,"_fonttest","std_gt")

def loadbw(p):
    a=np.asarray(Image.open(p).convert("L"),dtype=np.float32)
    return (a>128).astype(np.float32)

def centroid(a):
    ys,xs=np.nonzero(a)
    if len(xs)==0:return (0,0)
    return ys.mean(),xs.mean()

def proj(a):
    ys,xs=np.nonzero(a)
    if len(xs)==0: return None
    return int(ys.min()),int(ys.max()),int(xs.min()),int(xs.max()), int(ys.max()-ys.min()), int(xs.max()-xs.min())

# 隶书"一/二/三" 和楷书验证(这些是U+4E00系列)
tests=[
    ("li","U+04E00",  "一"),
    ("li","U+04E03",  "七"),
    ("kai","U+0342D","㐭"),
]
for book,fn,ch in tests:
    sp=os.path.join(STD,book,fn+".png"); gp=os.path.join(GTD,book,fn+".png")
    if not os.path.exists(sp) or not os.path.exists(gp):
        print(f"{book} {ch}: 缺文件"); continue
    s,t=loadbw(sp),loadbw(gp)
    s_p=proj(s); t_p=proj(t)
    print(f"\n{book} 字[{ch}] {fn}")
    print(f"  标准: 墨={int(s.sum())} 行范围[{s_p[0]}-{s_p[1]}]高{s_p[4]} 列范围[{s_p[2]}-{s_p[3]}]宽{s_p[5]}")
    print(f"  GT  : 墨={int(t.sum())} 行范围[{t_p[0]}-{t_p[1]}]高{t_p[4]} 列范围[{t_p[2]}-{t_p[3]}]宽{t_p[5]}")
    # 笔画连通分量数量(粗)
    from scipy.ndimage import label
    l1,n1=label(s); l2,n2=label(t)
    print(f"  连通分量: 标准={n1} GT={n2}")
