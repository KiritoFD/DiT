#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地生成 eval 单行对比图：每个样本一行，6 列：

  pred | GT | pred-canny | GT-canny | pred-skel | GT-skel

输入：eval_latest.png（远程拉回，每行左 pred 右 GT 并排，2*256 x N*256）。
骨架用 scikit-image 的标准 Zhang-Suen 细化（skeletonize），canny 用 cv2.Canny 边缘，
同一行内直接对比生成图与 GT 的结构。完全本地、轻量。

依赖：pip install scikit-image opencv-python-headless numpy pillow

用法:
  python make_eval_quad.py [eval_latest.png] [-o eval_quad.png]
"""
import os
import sys
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import skeletonize


CELL = 256
NCOLS = 6  # pred | GT | pred-canny | GT-canny | pred-skel | GT-skel


def _to_gray(pil_img):
    return np.asarray(pil_img.convert("L"), dtype=np.uint8)


def canny_edges(pil_img):
    gray = _to_gray(pil_img)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return Image.fromarray(edges).convert("RGB")


def skeleton(pil_img):
    """标准骨架：二值化（墨迹=亮区）→ skimage 细化（Zhang-Suen）→ 单像素骨架。"""
    gray = _to_gray(pil_img)
    ink = gray > 90          # 墨迹 mask（白底黑字取亮区笔画）
    skel = skeletonize(ink)  # 正确的中轴细化
    return Image.fromarray((skel * 255).astype(np.uint8)).convert("RGB")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "eval_latest.png")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "eval_quad.png")
    if not os.path.exists(src):
        print(f"[quad-local] 找不到 {src}，跳过")
        return 1

    img = Image.open(src).convert("RGB")
    w, h = img.size
    n_rows = h // CELL
    cols = w // CELL
    if cols < 2:
        print(f"[quad-local] 意外尺寸 {w}x{h}（预期每行 2 格），跳过")
        return 1

    canvas = Image.new("RGB", (CELL * NCOLS, CELL * n_rows), (20, 20, 20))
    for r in range(n_rows):
        pred = img.crop((0, r * CELL, CELL, (r + 1) * CELL))
        gt = img.crop((CELL, r * CELL, 2 * CELL, (r + 1) * CELL))
        cells = [pred, gt, canny_edges(pred), canny_edges(gt), skeleton(pred), skeleton(gt)]
        for c_i, cell in enumerate(cells):
            canvas.paste(cell, (c_i * CELL, r * CELL))
    canvas.save(out)
    print(f"[quad-local] {n_rows} 样本（每样本一行 6 列）-> {out} "
          f"(pred | GT | pred-canny | GT-canny | pred-skel | GT-skel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
