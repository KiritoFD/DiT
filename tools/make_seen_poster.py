#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Seen samples poster: 单独一张，展示训练集样本的生成结果 vs GT。
布局（时间升序向下）：
  顶部：实验名
  每个 seen step 行：标签行(step) + 图片行(gen|canny|skel, 5 样本)
  GT 行：对映最后 step 的 ground truth
底部注释
"""
import os, re, sys, glob, json, argparse, datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from skimage.morphology import skeletonize

CELL = 224
GAP = 6
BG = (15, 17, 22)
CELL_BG = (40, 40, 40)
GT_BG = (50, 50, 50)
GRID = (70, 78, 90)
HEADER_H = 56
LABEL_H = 56
LABEL_FONT = 48


def _step_key(d):
    m = re.search(r"step(\d+)", os.path.basename(d))
    return int(m.group(1)) if m else 0


def _complete_step_dirs(base):
    if not base or not os.path.isdir(base):
        return []
    out = []
    for d in sorted(glob.glob(os.path.join(base, "step*")), key=_step_key):
        if os.path.exists(os.path.join(d, "samples.json")):
            out.append(d)
    return out


def _gray_bt601(pil_img):
    a = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def canny_edges(pil_img):
    gray = _gray_bt601(pil_img)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, -1, ky, borderType=cv2.BORDER_REFLECT)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return Image.fromarray((mag > 150).astype(np.uint8) * 255).convert("RGB")


def skeleton(pil_img):
    gray = _gray_bt601(pil_img)
    bin_b = (gray > 127).astype(np.uint8)
    if gray.mean() > 127:
        bin_b = 1 - bin_b
    sk = skeletonize(bin_b.astype(bool)).astype(np.uint8) * 255
    return Image.fromarray(sk).convert("RGB")


def _load_cell(path, default_bg=CELL_BG):
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    return Image.new("RGB", (CELL, CELL), default_bg)


def _paste_cells(canvas, img, x, y):
    canvas.paste(img, (x, y))
    canvas.paste(canny_edges(img), (x + CELL, y))
    canvas.paste(skeleton(img), (x + 2 * CELL, y))


def _detect_n(step_dirs, cap=500):
    if not step_dirs:
        return 0
    d = step_dirs[0]
    n = 0
    for i in range(cap):
        if os.path.exists(os.path.join(d, f"sample{i}.png")):
            n = i + 1
        else:
            break
    return n


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seen-dir", required=True)
    ap.add_argument("--exp", default="")
    ap.add_argument("-o", "--out", default="seen_poster.png")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    seen_dirs = _complete_step_dirs(args.seen_dir)
    if not seen_dirs:
        print("No seen step dirs")
        return 1

    n_seen = _detect_n(seen_dirs)
    idxs = list(range(n_seen))

    n_cols = n_seen * 3
    W = CELL * n_cols + GAP * 2
    n_rows = len(seen_dirs) + 1  # steps + GT
    H = (GAP + HEADER_H
         + n_rows * (LABEL_H + CELL + GAP)
         + GAP + 30)

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    font = font_s = font_lab = None
    for _fp, _sz in [(r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17),
                     (r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14),
                     (r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", LABEL_FONT)]:
        try:
            _f = ImageFont.truetype(_fp, _sz)
            if font is None: font = _f
            elif font_s is None: font_s = _f
            elif font_lab is None: font_lab = _f
        except Exception:
            pass
    if font is None:
        font = font_s = ImageFont.load_default()
    if font_lab is None:
        font_lab = font

    y = GAP
    # Header
    draw.rectangle([0, y, W, y + HEADER_H], fill=(0, 0, 0))
    draw.text((GAP, y + (HEADER_H - 17) // 2),
              f"SEEN samples (train set, n={n_seen})",
              font=font, fill=(255, 200, 120))
    if args.exp:
        draw.text((W - GAP, y + (HEADER_H - 14) // 2), args.exp, anchor="ra",
                  font=font_s, fill=(150, 165, 190))
    y += HEADER_H

    def draw_label_row(y, text):
        draw.rectangle([0, y, W, y + LABEL_H], fill=(0, 0, 0))
        draw.line([(0, y + LABEL_H - 1), (W, y + LABEL_H - 1)], fill=GRID)
        draw.text((12, y + (LABEL_H - LABEL_FONT) // 2), text,
                  font=font_lab, fill=(255, 255, 255))
        return y + LABEL_H

    # Seen step rows
    for sd in seen_dirs:
        step = _step_key(sd)
        y = draw_label_row(y, f"SEEN STEP {step}")
        x = GAP
        for i in idxs:
            gen_path = os.path.join(sd, f"sample{i}.png")
            _paste_cells(canvas, _load_cell(gen_path), x, y)
            x += 3 * CELL
        draw.line([(0, y), (W, y)], fill=GRID)
        y += CELL + GAP

    # GT row
    y = draw_label_row(y, "GT (truth)")
    last_dir = seen_dirs[-1]
    x = GAP
    for i in idxs:
        gp = os.path.join(last_dir, f"gt{i}.png")
        _paste_cells(canvas, _load_cell(gp, GT_BG), x, y)
        x += 3 * CELL
    y += CELL + GAP

    # Footer
    draw.text((6, y + 4),
              f"{len(seen_dirs)} seen ckpt x {n_seen} samples | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
              font=font_s, fill=(90, 100, 120))

    canvas.save(args.out)
    print(f"[poster] {len(seen_dirs)} seen ckpt x {n_seen} -> {args.out} (exp={args.exp or 'none'})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
