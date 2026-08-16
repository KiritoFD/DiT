#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""改进指标: 标准skel vs GT骨架，质心对齐 + 可选缩放归一化后算拓扑 IoU / MSE。
判定"拓扑结构差不多 / MSE 差不多"是否成立。

关键: 不要求逐像素重合; 对齐后看宏观结构重叠度。
"""
import os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STD = os.path.join(HERE, "_fonttest", "std_skel")
GTD = os.path.join(HERE, "_fonttest", "std_gt")

def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) > 64

def translate_to_centroid(a):
    """把二值图质心移到图像中心(加 pad), 返回对齐后的二值图(同尺寸)。"""
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return a
    cy, cx = ys.mean(), xs.mean()
    H, W = a.shape
    dy = int(round(H / 2 - cy))
    dx = int(round(W / 2 - cx))
    from scipy.ndimage import shift
    return shift(a.astype(np.float32), shift=(dy, dx), order=0, mode='constant', cval=0) > 0.5

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)

def report(book, label):
    st = os.path.join(STD, book); gt = os.path.join(GTD, book)
    pairs = [(f, os.path.join(st, f), os.path.join(gt, f))
             for f in os.listdir(gt) if f.startswith("U+") and os.path.exists(os.path.join(st, f))]
    if not pairs:
        print(f"[{label}] 无配对"); return
    rows = []
    for f, sp, gp in pairs:
        s, g = load(sp), load(gp)
        if s.sum()==0 or g.sum()==0:
            continue
        sa, ga = translate_to_centroid(s), translate_to_centroid(g)
        iou_aligned = iou(sa, ga)          # 原尺寸对齐后 IoU
        iou_raw = iou(s, g)                # 未对齐 IoU(参考)
        # 低分辨率(32x32) IoU: 淡化微观差异
        from skimage.transform import resize
        s32 = resize(sa.astype(np.float32), (32,32), order=0, preserve_range=True) > 0.5
        g32 = resize(ga.astype(np.float32), (32,32), order=0, preserve_range=True) > 0.5
        iou32 = iou(s32, g32)
        rows.append((f, iou_raw, iou_aligned, iou32))
    I_raw=[r[1] for r in rows]; I_al=[r[2] for r in rows]; I32=[r[3] for r in rows]
    print(f"\n=== {label}: {len(rows)} 字 ===")
    print(f"  IoU 未对齐     : mean {np.mean(I_raw):.3f}")
    print(f"  IoU 质心对齐   : mean {np.mean(I_al):.3f}  median {np.median(I_al):.3f}")
    print(f"  IoU @32x32     : mean {np.mean(I32):.3f}  median {np.median(I32):.3f}  min {np.min(I32):.3f}")
    print(f"  >0.5(拓扑接近)数量: {(np.array(I32)>0.5).sum()}/{len(I32)}")
    worst = sorted(rows, key=lambda r: r[3])[:5]
    print(f"  最差5(32x32): " + ", ".join(r[0] for r in worst))

if __name__ == "__main__":
    report("kai", "楷书")
    report("li", "隶书")
