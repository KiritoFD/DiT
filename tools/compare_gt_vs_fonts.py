#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/compare_gt_vs_fonts.py — GT真迹拓本与名家数码字库结构、墨量与差分定量对比"""
import os
import sys
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SCRATCH = r"C:\Users\xy\.gemini\antigravity\brain\10d772c0-e343-4844-99e1-580c6676138e\scratch"
ARTIFACT_DIR = r"C:\Users\xy\.gemini\antigravity\brain\10d772c0-e343-4844-99e1-580c6676138e"
FONT_DIR = os.path.join(ROOT, "tools", "fonts")
CANDIDATE_DIR = os.path.join(ROOT, "tools", "fonts", "candidate_fonts")

SIZE = 256
BOX_FRAC = 0.88

_font_cache = {}


def get_font(path, size):
    key = (path, size)
    if key not in _font_cache:
        if os.path.exists(path):
            try:
                _font_cache[key] = ImageFont.truetype(path, size)
            except Exception:
                _font_cache[key] = None
        else:
            _font_cache[key] = None
    return _font_cache[key]


def render_glyph(ch, font_path, size=SIZE, box_frac=BOX_FRAC):
    large_size = size * 2
    font = get_font(font_path, int(large_size * 0.75))
    if font is None:
        return None
    img = Image.new("L", (large_size, large_size), 255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
    except Exception:
        return None
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 3 or h <= 3:
        return None
    x = (large_size - w) // 2 - bbox[0]
    y = (large_size - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)
    arr = np.asarray(img)
    ink = (arr < 230)
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1
    crop = img.crop((xmin, ymin, xmax, ymax))
    cw, ch_h = crop.size
    target = int(size * box_frac)
    scale = target / max(cw, ch_h)
    tw, th = max(1, int(cw * scale)), max(1, int(ch_h * scale))
    crop_resized = crop.resize((tw, th), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(crop_resized, ((size - tw) // 2, (size - th) // 2))
    return np.asarray(canvas)


def calc_metrics(gt_gray, font_gray):
    gt_ink = (gt_gray < 128)
    font_ink = (font_gray < 128)
    inter = np.logical_and(gt_ink, font_ink).sum()
    union = np.logical_or(gt_ink, font_ink).sum()
    iou = inter / max(1, union)
    sim, _ = ssim(gt_gray, font_gray, full=True)
    gt_ratio = float(gt_ink.mean())
    font_ratio = float(font_ink.mean())

    def get_centroid(mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return (128.0, 128.0)
        return (float(xs.mean()), float(ys.mean()))

    c_gt = get_centroid(gt_ink)
    c_fn = get_centroid(font_ink)
    c_dist = float(np.sqrt((c_gt[0] - c_fn[0])**2 + (c_gt[1] - c_fn[1])**2))
    return {
        "iou": float(iou),
        "ssim": float(sim),
        "gt_ratio": gt_ratio,
        "font_ratio": font_ratio,
        "c_dist": c_dist
    }


def create_diff_overlay(gt_gray, font_gray):
    gt_ink = (gt_gray < 128)
    font_ink = (font_gray < 128)
    h, w = gt_gray.shape
    overlay = np.full((h, w, 3), 255, dtype=np.uint8)
    overlay[gt_ink & ~font_ink] = [220, 50, 50]    # Red: GT only
    overlay[~gt_ink & font_ink] = [50, 90, 220]    # Blue: Font only
    overlay[gt_ink & font_ink] = [40, 160, 40]     # Green: Both (overlap)
    return overlay


def main():
    cases = [
        {
            "title": "颜真卿 楷书 《之》",
            "gt_file": "gt_yzq_zhi.png",
            "char": "之",
            "master_font": os.path.join(CANDIDATE_DIR, "yanti_shufa.ttf"),
            "master_label": "颜体字库",
            "std_font": os.path.join(FONT_DIR, "simkai.ttf"),
            "std_label": "中易楷体",
        },
        {
            "title": "颜真卿 楷书 《公》",
            "gt_file": "gt_yzq_gong.png",
            "char": "公",
            "master_font": os.path.join(CANDIDATE_DIR, "yanti_shufa.ttf"),
            "master_label": "颜体字库",
            "std_font": os.path.join(FONT_DIR, "simkai.ttf"),
            "std_label": "中易楷体",
        },
        {
            "title": "颜真卿 楷书 《德》",
            "gt_file": "gt_yzq_de.png",
            "char": "德",
            "master_font": os.path.join(CANDIDATE_DIR, "yanti_shufa.ttf"),
            "master_label": "颜体字库",
            "std_font": os.path.join(FONT_DIR, "simkai.ttf"),
            "std_label": "中易楷体",
        },
        {
            "title": "柳公权 楷书 《公》",
            "gt_file": "gt_lgq_gong.png",
            "char": "公",
            "master_font": os.path.join(CANDIDATE_DIR, "liugongquan_kaishu.ttf"),
            "master_label": "柳公权楷书",
            "std_font": os.path.join(FONT_DIR, "simkai.ttf"),
            "std_label": "中易楷体",
        },
        {
            "title": "柳公权 楷书 《之》",
            "gt_file": "gt_lgq_zhi.png",
            "char": "之",
            "master_font": os.path.join(CANDIDATE_DIR, "liugongquan_kaishu.ttf"),
            "master_label": "柳公权楷书",
            "std_font": os.path.join(FONT_DIR, "simkai.ttf"),
            "std_label": "中易楷体",
        },
        {
            "title": "王羲之 行书 《之》",
            "gt_file": "gt_wxz_zhi.png",
            "char": "之",
            "master_font": os.path.join(CANDIDATE_DIR, "ZhiMangXing-Regular.ttf"),
            "master_label": "志莽毛笔行书",
            "std_font": os.path.join(FONT_DIR, "STXINGKA.TTF"),
            "std_label": "华文行楷",
        },
        {
            "title": "邓石如 隶书 《公》",
            "gt_file": "gt_lishu_dsr_gong.png",
            "char": "公",
            "master_font": os.path.join(FONT_DIR, "SIMLI.TTF"),
            "master_label": "中易隶书",
            "std_font": os.path.join(FONT_DIR, "simkai.ttf"),
            "std_label": "标准楷体",
        },
        {
            "title": "赵之谦 隶书 《之》",
            "gt_file": "gt_lishu_zzq_zhi.png",
            "char": "之",
            "master_font": os.path.join(FONT_DIR, "SIMLI.TTF"),
            "master_label": "中易隶书",
            "std_font": os.path.join(FONT_DIR, "simkai.ttf"),
            "std_label": "标准楷体",
        }
    ]

    metrics_table = []
    rendered_rows = []

    for case in cases:
        gt_path = os.path.join(SCRATCH, case["gt_file"])
        gt_img = Image.open(gt_path).convert("L")
        gt_arr = np.asarray(gt_img)

        mf_arr = render_glyph(case["char"], case["master_font"])
        std_arr = render_glyph(case["char"], case["std_font"])

        m_mf = calc_metrics(gt_arr, mf_arr)
        m_std = calc_metrics(gt_arr, std_arr)

        metrics_table.append({
            "title": case["title"],
            "gt_ink": m_mf["gt_ratio"],
            "mf_label": case["master_label"],
            "mf_ink": m_mf["font_ratio"],
            "mf_iou": m_mf["iou"],
            "mf_ssim": m_mf["ssim"],
            "mf_cdist": m_mf["c_dist"],
            "std_label": case["std_label"],
            "std_ink": m_std["font_ratio"],
            "std_iou": m_std["iou"],
            "std_ssim": m_std["ssim"],
            "std_cdist": m_std["c_dist"],
        })

        diff_mf = create_diff_overlay(gt_arr, mf_arr)
        diff_std = create_diff_overlay(gt_arr, std_arr)

        rendered_rows.append({
            "case": case,
            "gt": gt_arr,
            "mf": mf_arr,
            "diff_mf": diff_mf,
            "std": std_arr,
            "diff_std": diff_std,
            "m_mf": m_mf,
            "m_std": m_std
        })

    print("\n=== GT真迹 vs 字库 定量对比表格 ===")
    print("| 样本真迹 | 真迹墨率 | 匹配名家字库 | 名家IoU | 名家SSIM | 质心偏 | 对照标准字库 | 对照IoU | 对照SSIM |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in metrics_table:
        print(f"| {r['title']} | {r['gt_ink']*100:.1f}% | {r['mf_label']} ({r['mf_ink']*100:.1f}%) | {r['mf_iou']*100:.1f}% | {r['mf_ssim']:.3f} | {r['mf_cdist']:.1f}px | {r['std_label']} ({r['std_ink']*100:.1f}%) | {r['std_iou']*100:.1f}% | {r['std_ssim']:.3f} |")

    # Render poster
    col_w = 256
    col_h = 256
    pad_x = 24
    pad_y = 52
    header_h = 130
    row_h = col_h + pad_y

    canvas_w = pad_x * 2 + col_w * 5 + pad_x * 4
    canvas_h = header_h + len(rendered_rows) * row_h + 30

    poster = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 250))
    draw = ImageDraw.Draw(poster)

    font_title = get_font("C:/Windows/Fonts/msyh.ttc", 26) or get_font("C:/Windows/Fonts/simhei.ttf", 26)
    font_col = get_font("C:/Windows/Fonts/msyh.ttc", 17) or get_font("C:/Windows/Fonts/simhei.ttf", 17)
    font_label = get_font("C:/Windows/Fonts/msyh.ttc", 14) or get_font("C:/Windows/Fonts/simhei.ttf", 14)

    draw.text((pad_x, 15), "真迹拓本 (GT) vs 名家数码字库 (Master Font) 结构几何与纹理量化对比", fill=(20, 20, 20), font=font_title)
    draw.text((pad_x, 54), "图例说明: [差分图] 绿色=真迹与字库完全重合 | 红色=真迹独有笔画(字库缺失/飞白) | 蓝色=字库多余笔画(几何外扩/标准化修饰)", fill=(90, 90, 90), font=font_col)

    headers = [
        "1. 真实历史真迹 (GT拓本)",
        "2. 专属名家字库 (Master Font)",
        "3. 真迹 vs 名家字库差分图",
        "4. 通用标准字库 (Standard Font)",
        "5. 真迹 vs 通用字库差分图"
    ]
    y_headers = 88
    for ci, h_text in enumerate(headers):
        cx = pad_x + ci * (col_w + pad_x)
        draw.text((cx, y_headers), h_text, fill=(30, 45, 80), font=font_col)

    y_start = header_h
    for ri, row_data in enumerate(rendered_rows):
        ry = y_start + ri * row_h
        case = row_data["case"]

        badge_text = f"【{case['title']}】 (名家IoU: {row_data['m_mf']['iou']*100:.1f}%, SSIM: {row_data['m_mf']['ssim']:.3f}, 质心偏移: {row_data['m_mf']['c_dist']:.1f}px)"
        draw.text((pad_x, ry - 18), badge_text, fill=(0, 65, 140), font=font_label)

        imgs = [
            Image.fromarray(row_data["gt"]).convert("RGB"),
            Image.fromarray(row_data["mf"]).convert("RGB"),
            Image.fromarray(row_data["diff_mf"]),
            Image.fromarray(row_data["std"]).convert("RGB"),
            Image.fromarray(row_data["diff_std"])
        ]

        sublabels = [
            f"真迹: 墨率={row_data['m_mf']['gt_ratio']*100:.1f}%",
            f"{case['master_label']}: 墨率={row_data['m_mf']['font_ratio']*100:.1f}%",
            f"重合IoU: {row_data['m_mf']['iou']*100:.1f}%",
            f"{case['std_label']}: 墨率={row_data['m_std']['font_ratio']*100:.1f}%",
            f"重合IoU: {row_data['m_std']['iou']*100:.1f}%"
        ]

        for ci, (im, slab) in enumerate(zip(imgs, sublabels)):
            cx = pad_x + ci * (col_w + pad_x)
            poster.paste(im, (cx, ry))
            draw.rectangle([cx - 1, ry - 1, cx + col_w, ry + col_h], outline=(205, 210, 215), width=1)
            draw.text((cx + 5, ry + col_h + 3), slab, fill=(70, 70, 70), font=font_label)

    out_poster_path = os.path.join(SCRATCH, "gt_vs_font_poster.png")
    poster.save(out_poster_path)
    print(f"\nPoster saved to: {out_poster_path}")

    art_poster_path = os.path.join(ARTIFACT_DIR, "gt_vs_font_poster.png")
    shutil.copyfile(out_poster_path, art_poster_path)
    print(f"Artifact poster copied to: {art_poster_path}")


if __name__ == "__main__":
    main()
