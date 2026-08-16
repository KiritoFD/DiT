#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比: 标准字 skel (楷/隶字体渲染) vs 数据 GT skeleton。
指标: 覆盖率(标准骨架命中GT) / precision(GT命中标准) / 像素差。
判断标准字库作为条件是否可接受。

读:
  _fonttest/std_skel/{kai,li}/U+XXXXX.png  (本脚本生成的标准字骨架)
  _fonttest/std_gt/{kai,li}/U+XXXXX.png    (远程拉回的数据GT骨架)
"""
import os, sys, glob
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
STD = os.path.join(HERE, "_fonttest", "std_skel")
GT = os.path.join(HERE, "_fonttest", "std_gt")

def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)

def report(book, label):
    std_dir = os.path.join(STD, book)
    gt_dir = os.path.join(GT, book)
    if not os.path.isdir(std_dir) or not os.path.isdir(gt_dir):
        print(f"[{label}] 缺目录 std/gt")
        return
    pairs = []
    for f in os.listdir(gt_dir):
        if not f.startswith("U+") or not f.endswith(".png"):
            continue
        gp = os.path.join(gt_dir, f)
        sp = os.path.join(std_dir, f)
        if os.path.exists(sp):
            pairs.append((f, sp, gp))
    if not pairs:
        print(f"[{label}] 无配对")
        return
    rows = []
    for f, sp, gp in pairs:
        s = load(sp); g = load(gp)
        # skel 图像素值 0/255; 转二值 1/0
        sb = (s > 128)          # 标准骨架
        gb = (g > 64)           # GT 骨架(降采样后阈值)
        inter = np.logical_and(sb, gb).sum()
        cov = inter / max(gb.sum(), 1)   # 标准骨架覆盖了GT多少
        prec = inter / max(sb.sum(), 1)  # 标准骨架里多少是GT的
        # 像素级 MSE(归一化)
        sn = s/s.max() if s.max()>0 else s
        gn = g/g.max() if g.max()>0 else g
        mse = float(np.mean((sn-gn)**2))
        rows.append((f, cov, prec, mse))
    covs = [r[1] for r in rows]; precs=[r[2] for r in rows]; mses=[r[3] for r in rows]
    print(f"\n=== {label}: {len(rows)} 字对比 ===")
    print(f"  标准skel→GT覆盖率: mean {np.mean(covs):.3f}  median {np.median(covs):.3f}  p10 {np.percentile(covs,10):.3f}")
    print(f"  标准skel precision : mean {np.mean(precs):.3f}  min {np.min(precs):.3f}")
    print(f"  像素MSE(归一化)     : mean {np.mean(mses):.4f}")
    print(f"  差 5 字: " + ", ".join(r[0] for r in sorted(rows,key=lambda r:r[1])[:5]))
    print(f"  good 5 字: " + ", ".join(r[0] for r in sorted(rows,key=lambda r:r[1],reverse=True)[:5]))

if __name__ == "__main__":
    report("kai", "楷书(标准楷体 vs 数据GT)")
    report("li",  "隶书(标准隶书 vs 数据GT)")
