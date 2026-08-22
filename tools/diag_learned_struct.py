# -*- coding: utf-8 -*-
"""量化诊断：生成图 vs GT 图 vs 噪声，看模型到底学到什么。

对每对 (sample_i.png, gt_i.png):
  - 灰度/L2 距离 (per-pixel)
  - SSIM (图像级)
  - 骨架覆盖率：min(gen_ink, gt_ink) / gt_ink   (生成墨迹有多少落在GT笔画区)
  - 骨架 precision/recall
对照：把纯随机噪声图也算同样指标，作为最低参考（若生成≈噪声就是没学会）。
"""
import os, sys, glob
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity as ssim

CELL = 256

def gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32)

def normalize(a):
    return (a - a.min()) / max(a.max() - a.min(), 1e-6)

def skel_mask(a):
    # 墨迹=暗色(书法白底黑字)，取 <128 为墨
    ink = a < 128
    # 反转给 skeletonize 用（骨架把亮区当物体）
    sk = skeletonize(ink)
    return sk

def report(name):
    g = gray(f"{name}_gt0.png")
    s = gray(f"{name}_sample0.png")
    # 归一化对比
    g_n = normalize(g); s_n = normalize(s)
    l2 = float(np.mean((s_n - g_n)**2))
    ss = float(ssim(s_n, g_n, data_range=1.0))
    gsk = skel_mask(g); ssk = skel_mask(s)
    inter = float(np.logical_and(ssk, gsk).sum())
    gsum = float(gsk.sum()) + 1e-6
    ssum = float(ssk.sum()) + 1e-6
    cover = inter / gsum       # 生成骨架命中GT骨架的比例
    prec = inter / ssum        # 生成骨架里有多少是GT的
    # 生成图本身的"结构度"：墨迹像素占比
    ink_frac = float((s < 128).mean())
    print(f"  {name}: L2(g,s)={l2*100:.2f}% SSIM={ss:.3f} | 骨架覆盖(gen→gt)={cover:.3f} prec={prec:.3f} | 生成墨占比={ink_frac:.3f}")
    return dict(l2=l2, ssim=ss, cover=cover, prec=prec, ink=ink_frac)

def rnd_reference():
    rng = np.random.RandomState(0)
    s = rng.rand(CELL, CELL) * 255
    return s

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "_cs3_check"
    print(f"=== 诊断目录 {base} ===")
    # 生成 vs GT：base 目录里是 sample0..4 / gt0..4
    all_r = []
    for i in range(5):
        sp = os.path.join(base, f"sample{i}.png")
        gp = os.path.join(base, f"gt{i}.png")
        if not os.path.exists(sp) or not os.path.exists(gp):
            continue
        r = _report_pair(sp, gp, f"sample{i}")
        all_r.append(r)
    # 随机噪声参考（用 gt0）
    g = gray(os.path.join(base, "gt0.png"))
    rnd = rnd_reference()
    g_n = normalize(g); r_n = normalize(rnd)
    l2r = float(np.mean((r_n-g_n)**2))
    ssr = float(ssim(r_n, g_n, data_range=1.0))
    gsk = skel_mask(g); rsk = skel_mask(rnd)
    intr = float(np.logical_and(rsk,gsk).sum()); gsumr=float(gsk.sum())+1e-6
    print(f"  随机噪声参考: L2={l2r*100:.2f}% SSIM={ssr:.3f} 骨架覆盖={intr/gsumr:.3f}")
    if all_r:
        am = np.mean([r['cover'] for r in all_r])
        print(f"\n结论: 平均骨架覆盖={am:.3f}；若 ≲ 随机噪声 => 模型没学到结构，只学了黑白")

def _report_pair(sp, gp, name):
    g = gray(gp); s = gray(sp)
    g_n = normalize(g); s_n = normalize(s)
    l2 = float(np.mean((s_n - g_n)**2))
    ss = float(ssim(s_n, g_n, data_range=1.0))
    gsk = skel_mask(g); ssk = skel_mask(s)
    inter = float(np.logical_and(ssk, gsk).sum())
    gsum = float(gsk.sum()) + 1e-6
    ssum = float(ssk.sum()) + 1e-6
    cover = inter / gsum
    prec = inter / ssum
    ink_frac = float((s < 128).mean())
    print(f"  {name}: L2(g,s)={l2*100:.2f}% SSIM={ss:.3f} | 骨架覆盖={cover:.3f} prec={prec:.3f} | 生成墨占比={ink_frac:.3f}")
    return dict(l2=l2, ssim=ss, cover=cover, prec=prec, ink=ink_frac)

if __name__ == "__main__":
    main()
