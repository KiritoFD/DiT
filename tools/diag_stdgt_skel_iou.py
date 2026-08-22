#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比: 标准字库渲染 skel vs GT skel 骨架重合度。
对 show5 5个字: 用对应字体(楷simkai/隶SIMLI)渲染 -> skel, 与远程GT真实骨架对比。
判断简体/繁体/异体导致的字形差异有多大。
"""
import os, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 256
FONT = {
    "楷": r"C:\Windows\Fonts\simkai.ttf",
    "隶": r"C:\Windows\Fonts\SIMLI.TTF",
}

def render_glyph_skeleton(ch, font_path):
    f = ImageFont.truetype(font_path, int(SIZE*0.86))
    img = Image.new("L", (SIZE, SIZE), 255)
    d = ImageDraw.Draw(img)
    d.text((SIZE*0.02, SIZE*0.02), ch, font=f, fill=0)
    g = np.asarray(img, dtype=np.float32)
    lo, hi = g.min(), g.max()
    if hi - lo < 1:
        return np.zeros((SIZE, SIZE), bool)
    g = (g - lo) / (hi - lo)
    ink = g < 128/255
    return skeletonize(ink)

def load_gt_skel(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return (a > 64)

def cov(s, t):
    inter = np.logical_and(s, t).sum()
    return inter / max(t.sum(), 1)

rows = list(csv.DictReader(open(os.path.join(HERE, "show5_eval.csv"), encoding="utf-8")))[:5]
print("=== 标准字库 skel vs GT skel 骨架重合度 ===")
for i, r in enumerate(rows):
    ch = r["character"]; script = r["script"]
    id_ = os.path.basename(r["image_path"])[:-4]
    fp = FONT.get(script)
    std = render_glyph_skeleton(ch, fp) if fp else np.zeros((SIZE, SIZE), bool)
    gt = load_gt_skel(os.path.join(HERE, "remote_gt", "skel", f"{id_}.png"))
    # 质心对齐
    def centroid(m):
        ys, xs = np.nonzero(m)
        return (ys.mean(), xs.mean()) if len(xs) else (0,0)
    from scipy.ndimage import shift
    cy_s, cx_s = centroid(std); cy_g, cx_g = centroid(gt)
    std_a = shift(std, shift=(SIZE/2-cy_s, SIZE/2-cx_s), order=0, mode='constant', cval=0)
    gt_a = shift(gt, shift=(SIZE/2-cy_g, SIZE/2-cx_g), order=0, mode='constant', cval=0)
    c_raw = cov(std, gt)
    c_al = cov(std_a, gt_a)
    # 标准字自己的 skel 量
    std_ink = std.sum()/256/256
    print(f"[{i}] {ch}(U+{ord(ch):04X}) {script} id={id_}")
    print(f"    标准skel量={std_ink:.3f} GTskel量={gt.sum()/65536:.3f}")
    print(f"    重合度(未对齐)={c_raw:.3f} 重合度(质心对齐)={c_al:.3f}")
