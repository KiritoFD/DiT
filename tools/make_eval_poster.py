#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地生成 eval 历史取样海报（与远程 GT 的 canny/skel 完全一致的算法）。

布局（一行 = 一个 ckpt，时间升序向下延伸，永不爆掉）：
  每行：该 step 的全部样本，每样本占 3 格 = sample | canny | skel
  列数 = 每 step 样本数 × 3（5 样本 -> 15 格）
  最后一行：GT（gt-img | gt-canny | gt-skel），只用一行。

算法（复刻远程 gen_canny_skel.py，保证与训练监督完全一致）：
  canny : 灰度 Rec.601(0.299R+0.587G+0.114B) -> Sobel 幅值 sqrt(gx^2+gy^2) > 150 -> 255
  skel  : 灰度 >127 二值化，均值>127 取反（适配白/黑底）-> Zhang-Suen 细化 -> 255
GT 行的 canny/skel 直接用远程 final_canny/final_skeleton 真值图（不含计算）。

数据源（远程拉回本地）：
  生成图 : remote_eval_samples/<exp>/.../eval_samples/stepXXXXXX/sample{i}.png
  GT真值  : remote_gt/{id}.png   (id=240699 等，来自 final_canny/final_skeleton)

用法:
  python make_eval_poster.py <eval_samples_local_dir> --gt-dir <remote_gt_dir> \
      [-o eval_poster.png] [--show5-csv 5script/eval_strata/clean_unseen_triple_100.csv]
"""
import os
import re
import sys
import csv
import glob
import numpy as np
from PIL import Image, ImageDraw
import cv2
from skimage.morphology import skeletonize

CELL = 224
GAP = 6


def _step_key(d):
    m = re.search(r"step(\d+)", os.path.basename(d))
    return int(m.group(1)) if m else 0


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


def _load_cell(path, default_bg=(40, 40, 40)):
    if os.path.exists(path):
        return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    return Image.new("RGB", (CELL, CELL), default_bg)


def _find_show5_ids(show5_csv):
    """固定 show5：取 eval_csv 前 5 行的图片 id（顺序固定）。"""
    if not show5_csv or not os.path.exists(show5_csv):
        return None
    rows = list(csv.DictReader(open(show5_csv, encoding="utf-8")))[:5]
    return [os.path.basename(r["image_path"])[:-4] for r in rows]


def _find_show5_meta(show5_csv):
    """返回 [(img_id, char, book_key), ...]，book_key 如 kai/li（用于定位标准字形图）。"""
    if not show5_csv or not os.path.exists(show5_csv):
        return []
    rows = list(csv.DictReader(open(show5_csv, encoding="utf-8")))[:5]
    out = []
    for r in rows:
        img_id = os.path.basename(r["image_path"])[:-4]
        char = r.get("character", "")
        key = r.get("std_glyph_key", "")
        book = key.split("/")[0] if "/" in key else ""
        out.append((img_id, char, book))
    return out


# 本地字体路径（无远程依赖）：楷=simkai, 隶=SIMLI
STD_FONT = {
    "kai": r"C:\Windows\Fonts\simkai.ttf",
    "li":  r"C:\Windows\Fonts\SIMLI.TTF",
}


def _render_std_font(char, book, size=256):
    """本地即时渲染标准字形图（白底黑字），无需预生成图/远程。"""
    from PIL import ImageFont, ImageDraw
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
    args = sys.argv[1:]
    if not args:
        print("用法: python make_eval_poster.py <eval_samples_local_dir> [--gt-dir DIRECTORY] [-o out.png]")
        return 1
    base = args[0]
    out = "eval_poster.png"
    gt_dir = None
    show5_csv = None
    i = 1
    while i < len(args):
        if args[i] == "-o" and i + 1 < len(args):
            out = args[i + 1]; i += 2
        elif args[i] == "--gt-dir" and i + 1 < len(args):
            gt_dir = args[i + 1]; i += 2
        elif args[i] == "--show5-csv" and i + 1 < len(args):
            show5_csv = args[i + 1]; i += 2
        else:
            out = args[i]; i += 1

    step_dirs = sorted(glob.glob(os.path.join(base, "step*")), key=_step_key)
    if not step_dirs:
        print(f"[poster] {base} 下没有 step 目录")
        return 1
    steps = [_step_key(d) for d in step_dirs]
    n_samples = max(len([f for f in os.listdir(d) if re.match(r"sample\d+\.png$", f)])
                    for d in step_dirs) or 1

    n_cols = n_samples * 3
    n_rows = len(step_dirs) + 2 + 1   # ckpt 行 + GT 行 + 标准字形行 + 顶部
    H = GAP + n_rows * (CELL + GAP)
    canvas = Image.new("RGB", (CELL * n_cols + GAP * 2, H), (15, 17, 22))
    draw = ImageDraw.Draw(canvas)

    y = GAP
    for ri, d in enumerate(step_dirs):
        x = GAP
        for i in range(n_samples):
            sp = os.path.join(d, f"sample{i}.png")
            img = _load_cell(sp)
            ce = canny_edges(img)
            sk = skeleton(img)
            canvas.paste(img, (x, y)); canvas.paste(ce, (x + CELL, y)); canvas.paste(sk, (x + 2 * CELL, y))
            x += 3 * CELL
        draw.text((4, y + CELL // 2), f"step{steps[ri]}", fill=(180, 190, 205))
        y += CELL + GAP

    # GT 行：优先用远程 GT 真值（final_canny/final_skeleton），否则本地对 gt 图算
    gt_y = y
    x = GAP
    show5 = _find_show5_ids(show5_csv)
    if gt_dir and show5:
        # GT 行用远程真值图：img + canny_gt + skel_gt
        gt_img_dir = None
        for d in step_dirs:
            if os.path.exists(os.path.join(d, "gt0.png")):
                gt_img_dir = d
                break
        for i in range(n_samples):
            pid = show5[i] if i < len(show5) else None
            gimg = _load_cell(os.path.join(gt_img_dir, f"gt{i}.png"), (50, 50, 50)) if gt_img_dir else Image.new("RGB", (CELL, CELL), (50, 50, 50))
            gce = _load_cell(os.path.join(gt_dir, "canny", f"{pid}.png"), (50, 50, 50)) if pid else Image.new("RGB", (CELL, CELL), (50, 50, 50))
            gsk = _load_cell(os.path.join(gt_dir, "skel", f"{pid}.png"), (50, 50, 50)) if pid else Image.new("RGB", (CELL, CELL), (50, 50, 50))
            canvas.paste(gimg, (x, gt_y)); canvas.paste(gce, (x + CELL, gt_y)); canvas.paste(gsk, (x + 2 * CELL, gt_y))
            x += 3 * CELL
    else:
        # 回退：对 gt 图用同一算法算（可能与远程有细微灰度差异，但结构一致）
        gt_dir = step_dirs[-1]
        for i in range(n_samples):
            gp = os.path.join(gt_dir, f"gt{i}.png")
            img = _load_cell(gp, (50, 50, 50))
            canvas.paste(img, (x, gt_y)); canvas.paste(canny_edges(img), (x + CELL, gt_y)); canvas.paste(skeleton(img), (x + 2 * CELL, gt_y))
            x += 3 * CELL
    draw.text((4, gt_y + CELL // 2), "GT", fill=(120, 230, 150))

    # 标准字形行（STD）：本地即时渲染对应 char 的标准字形（img | canny | skel），和 GT 对比
    std_y = gt_y + CELL + GAP
    metas = _find_show5_meta(show5_csv)
    x = GAP
    for i in range(n_samples):
        char_ = metas[i][1] if i < len(metas) else ""
        book_ = metas[i][2] if i < len(metas) else ""
        simg = _render_std_font(char_, book_)  # 本地字体渲染，(256,256,3)白底黑字
        simg = simg.resize((CELL, CELL), Image.LANCZOS) if simg else Image.new("RGB", (CELL, CELL), (60, 40, 40))
        ce = canny_edges(simg); sk = skeleton(simg)
        canvas.paste(simg, (x, std_y)); canvas.paste(ce, (x + CELL, std_y)); canvas.paste(sk, (x + 2 * CELL, std_y))
        x += 3 * CELL
    draw.text((4, std_y + CELL // 2), "STD", fill=(240, 160, 80))

    canvas.save(out)
    print(f"[poster] {len(step_dirs)} ckpt x {n_samples} 样本 -> {out} (GT 用 {'远程真值' if (gt_dir and show5) else '本地计算'}, 含 STD 标准字形行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
