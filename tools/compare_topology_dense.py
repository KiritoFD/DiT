#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""骨架拓扑相似度: 标准skel vs GT骨架。
对细骨架线做"模糊/膨胀"使其成为可比密度场, 然后算对齐后的 IoU / SSIM / MSE。
用户标准: 拓扑结构差不多 或 MSE 差不多即可接受。
"""
import os, glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
STD = os.path.join(HERE, "_fonttest", "std_skel")
GTD = os.path.join(HERE, "_fonttest", "std_gt")

def loadbw(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    return (a > 128).astype(np.float32)   # 骨架=白(1), 背景=黑(0)

def centroid_align(a):
    ys, xs = np.nonzero(a)
    if len(xs)==0: return a
    cy, cx = ys.mean(), xs.mean()
    H, W = a.shape
    from scipy.ndimage import shift
    return shift(a, shift=(H/2-cy, W/2-cx), order=0, mode='constant', cval=0)

def dense(a, sigma=4.0, size=64):
    """把骨架线模糊成密度场并降采样到 size×size。"""
    d = gaussian_filter(a, sigma=sigma)
    from skimage.transform import resize
    return resize(d, (size,size), order=1, mode='constant', preserve_range=True)

def report(book, label):
    st, gt = os.path.join(STD,book), os.path.join(GTD,book)
    pairs=[(f,os.path.join(st,f),os.path.join(gt,f)) for f in os.listdir(gt)
            if f.startswith("U+") and os.path.exists(os.path.join(st,f))]
    rows=[]
    for f,sp,gp in pairs:
        s,g=loadbw(sp),loadbw(gp)
        if s.sum()<10 or g.sum()<10: continue
        sa,ga=centroid_align(s),centroid_align(g)
        sd,gd=dense(sa),dense(ga)
        # 密度场 IoU (阈值>0.5 视为笔画)
        sb=sd>0.5; gb=gd>0.5
        iou = np.logical_and(sb,gb).sum()/max(np.logical_or(sb,gb).sum(),1)
        # 归一化 SSIM / MSE (密度场 0..1)
        ssn = sd/sd.max() if sd.max()>0 else sd
        gsn = gd/gd.max() if gd.max()>0 else gd
        ss = ssim(ssn, gsn, data_range=1.0)
        mse = float(np.mean((ssn-gsn)**2))
        rows.append((f,iou,ss,mse))
    I=[r[1] for r in rows]; SS=[r[2] for r in rows]; MS=[r[3] for r in rows]
    print(f"\n=== {label}: {len(rows)} 字 (骨架模糊成密度场后对齐比较) ===")
    print(f"  密度IoU   : mean {np.mean(I):.3f}  median {np.median(I):.3f}  min {np.min(I):.3f}")
    print(f"  SSIM(规范化): mean {np.mean(SS):.3f}  median {np.median(SS):.3f}")
    print(f"  MSE(规范化) : mean {np.mean(MS):.4f}  median {np.median(MS):.4f}")
    ok = sum(1 for i in I if i>0.3)
    print(f"  IoU>0.3(拓扑接近) : {ok}/{len(I)}")
    # 好的和差的示例
    worst=sorted(rows,key=lambda r:r[1])[:4]; best=sorted(rows,key=lambda r:r[1],reverse=True)[:4]
    print(f"  最差: " + ", ".join(r[0] for r in worst))
    print(f"  最好: " + ", ".join(r[0] for r in best))

if __name__=="__main__":
    report("kai","楷书")
    report("li","隶书")
