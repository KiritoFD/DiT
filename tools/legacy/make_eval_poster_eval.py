#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 make_eval_poster.py 的布局，支持 show5(eval 抽样) + seen5。
关键差异：不强制 step 交集。eval 用所有有 samples.json 的 step，
seen 只有 1 个 step，放在最下面一行（紧贴 GT 之前）。
布局（时间升序向下）：
  顶部：实验名
  eval 行（每个 step）：标签行(step+MSE/SSIM/LPIPS) + 图片行(gen, N 样本)
  seen 行（1 个 step）：标签行(SEEN step+MSE) + 图片行(gen, 5 样本)
  GT 行：eval GT + seen GT（对映最后 eval step + seen step）
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
    """每样本 3 格: gen | canny | skel"""
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
    ap.add_argument("--show5-dir", required=True)
    ap.add_argument("--seen5-dir", default=None)
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--exp", default="")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--step-stride", type=int, default=10)
    ap.add_argument("-o", "--out", default="poster.png")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    # eval metrics
    ev_map = {}
    if args.ckpt_dir and os.path.isdir(args.ckpt_dir):
        for f in sorted(glob.glob(os.path.join(args.ckpt_dir, "eval_auto_*.json"))):
            try:
                d = json.load(open(f))
                ev_map[int(d.get("step", 0))] = d
            except Exception:
                pass

    show_dirs = _complete_step_dirs(args.show5_dir)
    seen_dirs = _complete_step_dirs(args.seen5_dir)

    if args.step_stride > 1:
        show_dirs = show_dirs[::args.step_stride]
    # Always include last step
    if show_dirs and args.step_stride > 1:
        all_show = _complete_step_dirs(args.show5_dir)
        if all_show and all_show[-1] not in show_dirs:
            show_dirs.append(all_show[-1])

    if not show_dirs:
        print("No show5 step dirs")
        return 1

    n_show_total = _detect_n(show_dirs)
    n_show = min(args.n_samples, n_show_total) if n_show_total else args.n_samples
    # uniform sample indices
    if n_show_total > 0:
        idxs = [int(round(i * (n_show_total - 1) / max(n_show - 1, 1)))
                for i in range(n_show)]
    else:
        idxs = list(range(n_show))

    # Layout: header + eval steps (label+img) + GT row + footer
    n_cols = n_show * 3  # 3 cells per sample (gen|canny|skel)
    W = CELL * n_cols + GAP * 2
    n_rows = len(show_dirs) + 1  # +1 for GT
    H = (GAP + HEADER_H
         + n_rows * (LABEL_H + CELL + GAP)
         + GAP + 30)

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # fonts
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
              f"EVAL sampling (n={n_show}, idx={idxs[0]}..{idxs[-1]})",
              font=font, fill=(120, 200, 255))
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

    def draw_img_row(y, step_dir, idx_list, is_seen=False):
        x = GAP
        for i in idx_list:
            gen_path = os.path.join(step_dir, f"sample{i}.png")
            _paste_cells(canvas, _load_cell(gen_path), x, y)
            x += 3 * CELL
        draw.line([(0, y), (W, y)], fill=GRID)
        return y + CELL + GAP

    # Eval step rows
    for sd in show_dirs:
        step = _step_key(sd)
        parts = [f"STEP {step}"]
        ev = ev_map.get(step)
        if ev:
            for k in ("mse", "ssim", "lpips", "skel_iou"):
                v = ev.get(k)
                if v is not None:
                    parts.append(f"{k.upper()} {v:.3f}")
        y = draw_label_row(y, "    ".join(parts))
        y = draw_img_row(y, sd, idxs)

    # GT row
    gt_text = "GT (truth)"
    y = draw_label_row(y, gt_text)
    x = GAP
    # eval GTs (from last eval step)
    last_eval = show_dirs[-1]
    for i in idxs:
        gp = os.path.join(last_eval, f"gt{i}.png")
        _paste_cells(canvas, _load_cell(gp, GT_BG), x, y)
        x += 3 * CELL
    y += CELL + GAP

    # Footer
    draw.text((6, y + 4),
              f"{len(show_dirs)} eval ckpt x {n_show} samples | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
              font=font_s, fill=(90, 100, 120))

    canvas.save(args.out)
    print(f"[poster] {len(show_dirs)} eval -> {args.out} (exp={args.exp or 'none'})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
