#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_ckpt_poster.py — 单 ckpt 汇报海报（复用 render_poster_from_tree 的字体/配色轮子）。

布局（20 张生成图，5 列 × 4 图行，大字横幅行居中）：
    [可选 GT 细条: gt 缩略图 20 张]
    img row 1  (g 图 5 张)
    img row 2
    ★ BANNER: ckpt 名(特大字) + 方法 + 定量指标
    img row 3
    img row 4

样本选取：默认在可用 g{idx}.png 中均匀跨步取 20 张（不足则全取），
gt 与 g 严格同 idx 配对。输出单张 PNG。

用法:
  python tools/make_ckpt_poster.py --dir assets/milestones_samples/S0_85k \
      --name "S0@85k" --method "Writer Vector + Global AdaGN / 85k FM" \
      --metrics "OCR Top-1 81.29 | Top-5 89.47 | GT 89.47" \
      --out assets/milestones_posters/S0_85k.png
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image, ImageDraw, ImageFont
from render_poster_from_tree import find_font, BADGE_PALETTE


def numeric(files, prefix):
    out = []
    for p in files:
        m = re.search(prefix + r'(\d+)\.png$', os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return [p for _, p in sorted(out)]


def pick_indices(n_total, n_pick):
    if n_total <= n_pick:
        return list(range(n_total))
    return sorted({round(i * (n_total - 1) / (n_pick - 1)) for i in range(n_pick)})


def cell(img_path, size, label=None):
    """等比缩放居中到 size×size 白底；label 为左上角小标签(idx)。"""
    tile = Image.new("RGB", (size, size), "white")
    if img_path and os.path.exists(img_path):
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        s = min(size / w, size / h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        tile.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    else:
        d = ImageDraw.Draw(tile)
        d.text((size // 2, size // 2), "—", fill=(190, 190, 190), anchor="mm", font=find_font(size // 3))
    if label:
        d = ImageDraw.Draw(tile)
        f = find_font(max(14, size // 12))
        tw = d.textlength(label, font=f)
        d.rectangle([4, 4, 4 + tw + 10, 4 + size // 12 + 8], fill=(255, 255, 255))
        d.text((9, 6), label, fill=(90, 90, 90), font=f)
    return tile


def fit_font(draw, text, max_w, size, bold=False):
    """字号从 size 起步，缩到能放进 max_w 为止。"""
    f = find_font(size, bold=bold)
    while size > 10 and draw.textlength(text, font=f) > max_w:
        size = int(size * 0.92)
        f = find_font(size, bold=bold)
    return f


def banner(w, h, name, method, metrics, accent=True):
    pal = BADGE_PALETTE["accent" if accent else "default"]
    im = Image.new("RGB", (w, h), pal["box_bg"])
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=pal["border"], width=3)
    pad = w * 0.03
    f_name = fit_font(d, name, w - 2 * pad, int(h * 0.42), bold=True)
    f_sub = fit_font(d, method, w - 2 * pad, int(h * 0.17))
    f_met = fit_font(d, metrics, w - 2 * pad, int(h * 0.17))
    d.text((w // 2, int(h * 0.32)), name, fill=(15, 40, 90), font=f_name, anchor="mm")
    d.text((w // 2, int(h * 0.66)), method, fill=pal["text"], font=f_sub, anchor="mm")
    d.text((w // 2, int(h * 0.88)), metrics, fill=(120, 60, 10), font=f_met, anchor="mm")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='含 g{idx}.png（可选 gt{idx}.png）的样本目录')
    ap.add_argument('--name', required=True, help='ckpt 名（横幅大字）')
    ap.add_argument('--method', default='', help='方法一行')
    ap.add_argument('--metrics', default='', help='定量指标一行')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cols', type=int, default=5)
    ap.add_argument('--img-rows', type=int, default=4, help='横幅上下各一半的图行数')
    ap.add_argument('--cell', type=int, default=224, help='单元格边长 px')
    ap.add_argument('--gt-strip', action='store_true', help='顶部加 GT 细条（同 idx）')
    ap.add_argument('--idx-labels', action='store_true', help='单元格标注原始 idx')
    a = ap.parse_args()

    gens = numeric(os.listdir(a.dir), 'g')
    if not gens:
        sys.exit(f'no g*.png in {a.dir}')
    gens = [os.path.join(a.dir, p) for p in gens]
    gts = numeric(os.listdir(a.dir), 'gt')
    gt_by_idx = {}
    for p in gts:
        m = re.search(r'gt(\d+)\.png$', os.path.basename(p))
        gt_by_idx[int(m.group(1))] = os.path.join(a.dir, p)

    idxs = pick_indices(len(gens), a.cols * a.img_rows)
    cs = a.cell
    W = a.cols * cs
    H = int(cs * 1.9)
    gt_h = int(cs * 0.42) if a.gt_strip else 0
    Htot = gt_h + a.img_rows * cs + H

    poster = Image.new("RGB", (W, Htot), "white")
    if a.gt_strip:
        for c, gi in enumerate(idxs):
            poster.paste(cell(gt_by_idx.get(gi), gt_h), (c * cs, 0))
    half_up = a.img_rows // 2
    for r in range(a.img_rows):
        row = r if r < half_up else r + 1  # 中间一行让给横幅
        y = gt_h + row * cs
        for c, gi in enumerate(idxs):
            src = gens[gi] if gi < len(gens) else None
            label = str(gens.index(src)) if False else (str(re.search(r'g(\d+)', os.path.basename(src)).group(1)) if a.idx_labels and src else None)
            poster.paste(cell(src, cs, label), (c * cs, y))
    by = gt_h + half_up * cs
    poster.paste(banner(W, H, a.name, a.method, a.metrics), (0, by))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    poster.save(a.out)
    print('POSTER OK', a.out, poster.size)


if __name__ == '__main__':
    main()
