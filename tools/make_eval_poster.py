#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地生成 eval 历史取样海报（与远程 GT 的 canny/skel 完全一致的算法）。

布局（一行 = 一个 ckpt，时间升序向下延伸，永不爆掉）：
  每行：show5(5 个 unseen 样本) + seen5(5 个训练样本) = 10 样本，
        每样本占 3 格 = sample | canny | skel  ->  每行 30 格
  顶部一行：分组标签（SHOW5 / SEEN5，附样本序号+字形）
  最后一行：GT（show5 用远程真值图 / seen5 用 seen_samples 里的 gt），只一行。

算法（复刻远程 gen_canny_skel.py，保证与训练监督完全一致）：
  canny : 灰度 Rec.601(0.299R+0.587G+0.114B) -> Sobel 幅值 sqrt(gx^2+gy^2) > 150 -> 255
  skel  : 灰度 >127 二值化，均值>127 取反（适配白/黑底）-> Zhang-Suen 细化 -> 255
GT 行的 show5 canny/skel 直接用远程 final_canny/final_skeleton 真值图（不含计算）。

完整性约定（与 CPU eval 解耦，只渲染已完成 step）：
  只渲染带 samples.json 的 step 目录（eval 最后写 samples.json = 该 step 已画完）。
  show5 与 seen5 两侧都有 samples.json 的 step 才进海报，其余跳过。

数据源（远程拉回本地）：
  生成图(show5) : remote_eval_samples/<exp>/eval_samples/stepXXXXXX/sample{i}.png
  生成图(seen5) : remote_seen_samples/<exp>/seen_samples/stepXXXXXX/sample{i}.png
  GT真值(show5) : remote_gt/{id}.png  (id=240699 等，来自 final_canny/final_skeleton)
  GT真值(seen5) : seen_samples/stepXXXXXX/gt{i}.png

用法:
  python make_eval_poster.py --show5-dir <dir> --seen5-dir <dir> \
      [--gt-dir <remote_gt_dir>] [--show5-csv <csv>] [--seen5-csv <csv>] \
      [--exp <experiment>] [-o out.png]

  旧用法（仅 show5，5 列）仍兼容：
  python make_eval_poster.py <eval_samples_local_dir> [--gt-dir <dir>] [-o out.png]
"""
import os
import re
import sys
import csv
import glob
import argparse
import datetime
import json
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


def _step_key(d):
    m = re.search(r"step(\d+)", os.path.basename(d))
    return int(m.group(1)) if m else 0


def _complete_step_dirs(base):
    """只返回带 samples.json 的 step 目录（eval 写全样本后最后落 samples.json）。"""
    if not base or not os.path.isdir(base):
        return []
    out = []
    for d in sorted(glob.glob(os.path.join(base, "step*")), key=_step_key):
        if os.path.exists(os.path.join(d, "samples.json")):
            out.append(d)
    return out


def _gray_bt601(pil_img):
    """与远程一致：Rec.601 (0.299R+0.587G+0.114B)，返回 float32 数组 [0,255]。"""
    a = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def canny_edges(pil_img):
    """复刻远程：Sobel 幅值 >150 直接二值（非 cv2.Canny）。"""
    gray = _gray_bt601(pil_img)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, -1, ky, borderType=cv2.BORDER_REFLECT)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    bin_img = (mag > 150).astype(np.uint8) * 255
    return Image.fromarray(bin_img).convert("RGB")


def skeleton(pil_img):
    """复刻远程：灰度>127 二值，均值>127 取反，Zhang-Suen 细化。"""
    gray = _gray_bt601(pil_img)
    bin_b = (gray > 127).astype(np.uint8)
    if gray.mean() > 127:  # 白底黑字 -> 取反让墨迹=1
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


def _find_group_meta(csv_path, n=5):
    """返回 [(img_id, char), ...] 取 csv 前 n 行。"""
    if not csv_path or not os.path.exists(csv_path):
        return []
    out = []
    try:
        for r in list(csv.DictReader(open(csv_path, encoding="utf-8")))[:n]:
            img_id = os.path.basename(r.get("image_path", ""))[:-4]
            out.append((img_id, r.get("character", "")))
    except Exception:
        pass
    return out


# 本地字体路径（无远程依赖）：楷=simkai, 隶=SIMLI
STD_FONT = {
    "kai": r"C:\Windows\Fonts\simkai.ttf",
    "li":  r"C:\Windows\Fonts\SIMLI.TTF",
}


def _render_std_font(char, book, size=256):
    """本地即时渲染标准字形图（白底黑字），无需预生成图/远程。"""
    fp = STD_FONT.get(book)
    if not fp or not os.path.exists(fp) or not char:
        return None
    try:
        f = ImageFont.truetype(fp, int(size * 0.86))
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((size * 0.02, size * 0.02), char, font=f, fill=(0, 0, 0))
        return img
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--show5-dir")
    ap.add_argument("--seen5-dir")
    ap.add_argument("--gt-dir")
    ap.add_argument("--show5-csv")
    ap.add_argument("--seen5-csv")
    ap.add_argument("--exp", default="")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--eval-json-dir", default=None,
                    help="ckpt 目录含 eval_auto_*.json；每行 step 标签下追加 MSE/SSIM")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    # 可选: 读 ckpt 目录的 eval_auto_*.json -> step → (mse, ssim)
    ev_map = {}
    if args.eval_json_dir and os.path.isdir(args.eval_json_dir):
        import glob as _g
        for f in _g.glob(os.path.join(args.eval_json_dir, "eval_auto_*.json")):
            try:
                d = json.load(open(f))
                ev_map[int(d.get("step", 0))] = (d.get("mse"), d.get("ssim"))
            except Exception:
                pass

    show5_dir = args.show5_dir or (args.dirs[0] if args.dirs else None)
    seen5_dir = args.seen5_dir
    out = args.out or (f"eval_poster_{args.exp}.png" if args.exp else "eval_poster.png")

    show5_dirs = _complete_step_dirs(show5_dir)
    seen5_dirs = _complete_step_dirs(seen5_dir)
    if not show5_dirs and not seen5_dirs:
        print("[poster] 两侧都没有带 samples.json 的 step 目录（eval 未产出或未完成）")
        return 1

    # 行 = 两侧都完成的 step 交集（时间升序）；单侧时只显示那一侧
    if seen5_dirs:
        steps = sorted(set(_step_key(d) for d in show5_dirs) &
                       set(_step_key(d) for d in seen5_dirs))
        by_step = lambda D: {_step_key(d): d for d in D}  # noqa: E731
        show5_map, seen5_map = by_step(show5_dirs), by_step(seen5_dirs)
    else:
        steps = sorted(_step_key(d) for d in show5_dirs)
        show5_map = {_step_key(d): d for d in show5_dirs}
        seen5_map = {}
    if not steps:
        print("[poster] show5/seen5 step 无交集（可能一侧还在跑）")
        return 1

    n_show = 5
    n_seen = 5 if seen5_dirs else 0
    n_cols = (n_show + n_seen) * 3
    HEADER_H = 40
    n_rows = 1 + len(steps) + 1 + 1          # 顶部分组标签 + step 行 + GT 行 + 底部注释
    W = CELL * n_cols + GAP * 2
    H = GAP + HEADER_H + n_rows * (CELL + GAP)
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 17)
        font_s = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 14)
    except Exception:
        font = font_s = ImageFont.load_default()

    y = GAP

    # 顶部分组标签行
    gy = y + (HEADER_H - 20) // 2
    show5_meta = _find_group_meta(args.show5_csv)
    seen5_meta = _find_group_meta(args.seen5_csv)
    if n_seen:
        sx = GAP + 0 * 3 * CELL
        draw.text((sx + 6, gy), f"SHOW5 (unseen)", font=font, fill=(120, 200, 255))
        draw.text((sx + 6, gy + 18), " | ".join(
            f"#{i+1}{('·'+(show5_meta[i][1] if i < len(show5_meta) and show5_meta[i][1] else ''))}" for i in range(5)),
            font=font_s, fill=(150, 165, 190))
        sx = GAP + 5 * 3 * CELL
        draw.text((sx + 6, gy), f"SEEN5 (train)", font=font, fill=(255, 200, 120))
        draw.text((sx + 6, gy + 18), " | ".join(
            f"#{i+1}{('·'+(seen5_meta[i][1] if i < len(seen5_meta) and seen5_meta[i][1] else ''))}" for i in range(5)),
            font=font_s, fill=(150, 165, 190))
    else:
        sx = GAP + 0 * 3 * CELL
        draw.text((sx + 6, gy), "SHOW5 (unseen)", font=font, fill=(120, 200, 255))
        draw.text((sx + 6, gy + 18), " | ".join(
            f"#{i+1}{('·'+(show5_meta[i][1] if i < len(show5_meta) and show5_meta[i][1] else ''))}" for i in range(5)),
            font=font_s, fill=(150, 165, 190))
    if args.exp:
        draw.text((W - 6, 4), args.exp, anchor="ra", font=font_s, fill=(90, 100, 120))
    y += HEADER_H

    # step 行
    for step in steps:
        x = GAP
        for i in range(n_show):
            _paste_cells(canvas, _load_cell(os.path.join(show5_map[step], f"sample{i}.png")), x, y)
            x += 3 * CELL
        if n_seen:
            for i in range(n_seen):
                _paste_cells(canvas, _load_cell(os.path.join(seen5_map[step], f"sample{i}.png")), x, y)
                x += 3 * CELL
        draw.line([(0, y), (W, y)], fill=GRID)
        draw.line([(GAP + 5 * 3 * CELL, y), (GAP + 5 * 3 * CELL, y + CELL)], fill=GRID)
        # 每行左侧标签: step 号 + (可选)该 ckpt 的 eval MSE/SSIM
        draw.text((4, y + 2), f"step {step}", font=font_s, fill=(180, 190, 205))
        if step in ev_map:
            _m, _s = ev_map[step]
            _txt = f"MSE {_m:.3f}" if _m is not None else "MSE --"
            if _s is not None:
                _txt += f"\nSSIM {_s:.3f}"
            draw.multiline_text((4, y + 24), _txt, font=font_s, fill=(140, 150, 170))
        y += CELL + GAP

    # GT 行：show5 用远程真值图(gt_dir/{pid}.png)，seen5 用 seen_samples 里的 gt{i}.png
    gt_y = y
    x = GAP
    show5_ids = [i for i, _ in _find_group_meta(args.show5_csv)]
    for i in range(n_show):
        pid = show5_ids[i] if i < len(show5_ids) else None
        gp = os.path.join(args.gt_dir, f"{pid}.png") if (args.gt_dir and pid) else None
        if not gp or not os.path.exists(gp):
            gp = None
            for d in show5_dirs:
                if os.path.exists(os.path.join(d, f"gt{i}.png")):
                    gp = os.path.join(d, f"gt{i}.png")
                    break
        _paste_cells(canvas, _load_cell(gp, GT_BG), x, gt_y)
        x += 3 * CELL
    if n_seen:
        for i in range(n_seen):
            gp = os.path.join(seen5_map[steps[-1]], f"gt{i}.png")
            _paste_cells(canvas, _load_cell(gp, GT_BG), x, gt_y)
            x += 3 * CELL
    draw.text((4, gt_y + 2), "GT", font=font, fill=(120, 230, 150))
    if steps:
        draw.text((4, gt_y + 28), f"(真值, step {steps[-1]})", font=font_s, fill=(120, 230, 150))
    y += CELL + GAP

    # 底部注释
    draw.text((6, y + 4),
              f"{len(steps)} ckpt x {n_show + n_seen} 样本(img|canny|skel) | 生成 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
              font=font_s, fill=(90, 100, 120))

    canvas.save(out)
    print(f"[poster] {len(steps)} ckpt x {n_show + n_seen} 样本 -> {out} (exp={args.exp or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())