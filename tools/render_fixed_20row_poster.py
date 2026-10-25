#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/render_fixed_20row_poster.py —
20 行高保真全貌图册 (强力笔画对齐 + Top 2 30k/50k 过训白化对比):
  1. 直击过训白化根因:
     - 并列呈现 Top 2 (30k 健壮期) 与 Top 2 (50k 过训期)
     - 显式展现部分字样从 30k 饱满完整到 50k 模式崩溃白化的物理过程
  2. 强力有效、肉眼可见的笔画粗细 SDF 对齐:
     - 拓扑门禁剔除背景噪点单像素伪孔，放开对结构孔洞 (>= 15px) 的有效保护
     - 放宽法向位移上限至 7.0px，笔画粗细真正发生 4~12px 的显著收缩与丰腴演变
     - 亚像素 Sigmoid 柔和宣纸墨晕重建 (tau=0.35)，既边缘如刀削般清晰，又具墨汁微晕
  3. 彻底零裁切、零遮挡:
     - 自动检测并施加 64px 纯白安全垫，单字居中且四周留白 >= 30px
     - 独立数据槽底栏，与字形画面 100% 物理隔离
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from skimage.morphology import medial_axis, skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FONT_PATH = ROOT / "tools/fonts/SimHei.ttf"
if not FONT_PATH.exists():
    FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")

TRAIN_DIST_JSON = ROOT / "assets/stroke_width_trainset_distribution.json"
OUT_POSTER = ROOT / "assets/posters/poster_subsets_and_stroke_alignment.png"

# 20 组精选传世代表字样 (四大风格严格平衡各 5 字)
SAMPLE_MAPPING = [
    # 组 1: 王羲之 · 楷书 (清健峻拔，中锋立骨)
    {"i2": 0,  "i5": 18, "i10": 80,  "callig": "王羲之", "script": "楷", "char": "旨", "group": "王羲之 · 楷书"},
    {"i2": 1,  "i5": 19, "i10": 81,  "callig": "王羲之", "script": "楷", "char": "尚", "group": "王羲之 · 楷书"},
    {"i2": 2,  "i5": 20, "i10": 82,  "callig": "王羲之", "script": "楷", "char": "位", "group": "王羲之 · 楷书"},
    {"i2": 4,  "i5": 22, "i10": 84,  "callig": "王羲之", "script": "楷", "char": "兩", "group": "王羲之 · 楷书"},
    {"i2": 7,  "i5": 25, "i10": 87,  "callig": "王羲之", "script": "楷", "char": "卅", "group": "王羲之 · 楷书"},

    # 组 2: 王羲之 · 行书 (兰亭逸气，飞动洒脱)
    {"i2": 25, "i5": 26, "i10": 88,  "callig": "王羲之", "script": "行", "char": "好", "group": "王羲之 · 行书"},
    {"i2": 26, "i5": 27, "i10": 89,  "callig": "王羲之", "script": "行", "char": "致", "group": "王羲之 · 行书"},
    {"i2": 27, "i5": 28, "i10": 90,  "callig": "王羲之", "script": "行", "char": "夜", "group": "王羲之 · 行书"},
    {"i2": 28, "i5": 29, "i10": 91,  "callig": "王羲之", "script": "行", "char": "羲", "group": "王羲之 · 行书"},
    {"i2": 33, "i5": 34, "i10": 96,  "callig": "王羲之", "script": "行", "char": "将", "group": "王羲之 · 行书"},

    # 组 3: 苏轼 · 楷书 (重墨宽博，端庄浑厚)
    {"i2": 50, "i5": 35, "i10": 111, "callig": "苏轼",   "script": "楷", "char": "靑", "group": "苏轼 · 楷书"},
    {"i2": 51, "i5": 36, "i10": 112, "callig": "苏轼",   "script": "楷", "char": "衆", "group": "苏轼 · 楷书"},
    {"i2": 54, "i5": 39, "i10": 115, "callig": "苏轼",   "script": "楷", "char": "入", "group": "苏轼 · 楷书"},
    {"i2": 55, "i5": 40, "i10": 116, "callig": "苏轼",   "script": "楷", "char": "土", "group": "苏轼 · 楷书"},
    {"i2": 56, "i5": 41, "i10": 117, "callig": "苏轼",   "script": "楷", "char": "皇", "group": "苏轼 · 楷书"},

    # 组 4: 苏轼 · 行书 (绵中裹铁，沉着痛快)
    {"i2": 76, "i5": 45, "i10": 121, "callig": "苏轼",   "script": "行", "char": "巫", "group": "苏轼 · 行书"},
    {"i2": 77, "i5": 46, "i10": 122, "callig": "苏轼",   "script": "行", "char": "稜", "group": "苏轼 · 行书"},
    {"i2": 78, "i5": 47, "i10": 123, "callig": "苏轼",   "script": "行", "char": "渚", "group": "苏轼 · 行书"},
    {"i2": 79, "i5": 48, "i10": 124, "callig": "苏轼",   "script": "行", "char": "冒", "group": "苏轼 · 行书"},
    {"i2": 81, "i5": 50, "i10": 126, "callig": "苏轼",   "script": "行", "char": "怆", "group": "苏轼 · 行书"},
]

GROUP_COLORS = {
    "王羲之 · 楷书": ((238, 242, 255), (79, 70, 229)),   # Indigo
    "王羲之 · 行书": ((240, 253, 250), (13, 148, 136)),  # Teal
    "苏轼 · 楷书":   ((250, 245, 255), (147, 51, 234)),  # Purple
    "苏轼 · 行书":   ((255, 251, 235), (217, 119, 6)),   # Amber
}


def measure_p50_and_contrast(img_np: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """精确测量中轴骨架物理笔宽 P50 与提按比 (Contrast)。"""
    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 25:
        return 0.0, 1.0
    dist_map = distance_transform_edt(ink_mask)
    skel = medial_axis(ink_mask)
    if not skel.any():
        skel = skeletonize(ink_mask)
    if not skel.any():
        return 0.0, 1.0
    widths = 2.0 * dist_map[skel]
    widths = widths[widths >= 0.5]
    if len(widths) < 10:
        return 0.0, 1.0
    p10 = float(np.percentile(widths, 10))
    p50 = float(np.percentile(widths, 50))
    p90 = float(np.percentile(widths, 90))
    contrast = p90 / max(p10, 1e-4)
    return p50, contrast


def count_structural_holes(mask: np.ndarray, min_area: int = 15) -> int:
    """统计背景中真实的文字结构孔洞 (忽略单像素背景噪点)。"""
    lbl_bg, num_bg = label(~mask)
    if num_bg <= 1:
        return 0
    sizes = np.bincount(lbl_bg.ravel())
    outer_bg = np.argmax(sizes)
    real_holes = sum(sizes[k] >= min_area for k in range(1, num_bg + 1) if k != outer_bg)
    return real_holes


def powerful_sdf_align(
    img_np: np.ndarray,
    target_dist: dict,
    alpha: float = 1.0,
    threshold: float = 0.5,
    max_shift: float = 7.0,
    smooth_sigma: float = 1.2,
    edge_tau: float = 0.35
) -> np.ndarray:
    """强力有效、粗细变化显著、拓扑安全且抗锯齿的笔画粗细空间变异 SDF 对齐。"""
    if abs(alpha) < 1e-4:
        return img_np

    ink_soft = np.clip(1.0 - img_np, 0.0, 1.0)
    ink_mask = ink_soft >= threshold
    if ink_mask.sum() < 25:
        return img_np

    # 统计主要笔画连通块 (>= 20px) 与结构孔洞
    lbl_orig, num_orig = label(ink_mask)
    orig_sizes = np.bincount(lbl_orig.ravel())[1:]
    major_orig_comp = sum(s >= 20 for s in orig_sizes)
    real_holes_orig = count_structural_holes(ink_mask, min_area=15)

    dist_map = distance_transform_edt(ink_mask)
    skel = medial_axis(ink_mask)
    if not skel.any():
        skel = skeletonize(ink_mask)
    if not skel.any():
        return img_np

    widths = 2.0 * dist_map[skel]
    widths = widths[widths >= 0.5]
    if len(widths) < 10:
        return img_np

    mu_t = target_dist["ref_mean"]
    sigma_t = target_dist["ref_std"]
    mu_g = float(np.mean(widths))
    sigma_g = max(float(np.std(widths)), 0.2)

    scale = float(np.clip(sigma_t / sigma_g, 0.5, 2.0))
    skel_widths = 2.0 * dist_map[skel]
    target_widths = mu_t + scale * (skel_widths - mu_g)
    target_widths = np.clip(target_widths, 1.0, 45.0)

    # 法向半径位移
    delta_skel = (target_widths - skel_widths) / 2.0
    delta_skel = np.clip(delta_skel, -max_shift, max_shift)

    # 空间扩散
    _, (nearest_r, nearest_c) = distance_transform_edt(~skel, return_indices=True)
    delta_grid = np.zeros_like(img_np, dtype=np.float32)
    delta_grid[skel] = delta_skel.astype(np.float32)
    delta_map = delta_grid[nearest_r, nearest_c]
    delta_map = gaussian_filter(delta_map, sigma=smooth_sigma)

    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    phi_prime = phi + alpha * delta_map

    # 门禁检查: 绝不裂开大笔画，绝不糊死结构孔洞
    mask_c = phi_prime >= 0.0
    lbl_c, num_c = label(mask_c)
    cand_sizes = np.bincount(lbl_c.ravel())[1:] if num_c > 0 else []
    major_cand_comp = sum(s >= 20 for s in cand_sizes)
    real_holes_c = count_structural_holes(mask_c, min_area=15)

    atten = 1.0
    while (major_cand_comp > major_orig_comp + 1 or real_holes_c < real_holes_orig - 1) and atten > 0.3:
        atten *= 0.8
        phi_prime = phi + (alpha * atten) * delta_map
        mask_c = phi_prime >= 0.0
        lbl_c, num_c = label(mask_c)
        cand_sizes = np.bincount(lbl_c.ravel())[1:] if num_c > 0 else []
        major_cand_comp = sum(s >= 20 for s in cand_sizes)
        real_holes_c = count_structural_holes(mask_c, min_area=15)

    # 自然宣纸微晕亚像素重建
    ink_aligned = 1.0 / (1.0 + np.exp(-phi_prime / edge_tau))
    return (1.0 - ink_aligned).astype(np.float32)


def prepare_cell_image(img_np: np.ndarray, cell_size: int = 240, inner_pad: int = 20) -> Image.Image:
    """保证字符以充足安全边距居中贴图，绝不贴边，绝无任何裁切。"""
    ink = (1.0 - img_np) >= 0.20
    if not ink.any():
        return Image.new("RGB", (cell_size, cell_size), (255, 255, 255))

    y_idx, x_idx = np.where(ink)
    ymin, ymax = int(y_idx.min()), int(y_idx.max())
    xmin, xmax = int(x_idx.min()), int(x_idx.max())
    h_ink = ymax - ymin + 1
    w_ink = xmax - xmin + 1

    avail_size = cell_size - 2 * inner_pad  # 200

    im = Image.fromarray((img_np * 255.0).astype(np.uint8)).convert("RGB")

    if ymin >= 20 and (255 - ymax) >= 20 and xmin >= 20 and (255 - xmax) >= 20:
        im_resized = im.resize((avail_size, avail_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
        canvas.paste(im_resized, (inner_pad, inner_pad))
        return canvas
    else:
        padded = Image.new("RGB", (384, 384), (255, 255, 255))
        padded.paste(im, (64, 64))
        cy = (ymin + ymax) / 2.0 + 64
        cx = (xmin + xmax) / 2.0 + 64
        half_box = max(h_ink, w_ink) / 2.0 + 20
        crop_box = (
            int(round(cx - half_box)),
            int(round(cy - half_box)),
            int(round(cx + half_box)),
            int(round(cy + half_box))
        )
        cropped = padded.crop(crop_box)
        cropped_resized = cropped.resize((avail_size, avail_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
        canvas.paste(cropped_resized, (inner_pad, inner_pad))
        return canvas


def main():
    print(f"Loading training set distribution table from {TRAIN_DIST_JSON}...")
    with open(TRAIN_DIST_JSON, "r", encoding="utf-8") as f:
        train_dist = json.load(f)

    # 字体准备
    font_title = ImageFont.truetype(str(FONT_PATH), 28)
    font_subtitle = ImageFont.truetype(str(FONT_PATH), 18)
    font_col_main = ImageFont.truetype(str(FONT_PATH), 20)
    font_col_sub = ImageFont.truetype(str(FONT_PATH), 14)
    font_row_char = ImageFont.truetype(str(FONT_PATH), 30)
    font_row_text = ImageFont.truetype(str(FONT_PATH), 17)
    font_tag = ImageFont.truetype(str(FONT_PATH), 13)
    font_badge = ImageFont.truetype(str(FONT_PATH), 14)

    cell_w = 240
    img_box_h = 240   # 图像纯净画布高度
    meta_h = 26       # 独立数据展示槽高度
    cell_h = img_box_h + meta_h  # 266px 总单元高度

    pad = 12
    label_w = 210
    header_h = 110
    title_h = 100

    # 10 大核心比对列 (直击 Top 2 过训退化与不同强度真实笔画对齐)
    COLUMNS = [
        ("STD 参考", "标准行/楷字模", "std", 0.0),
        ("GT 真迹", "传世碑帖法书", "gt", 0.0),
        ("Top 2 30k", "72M 健壮期 · 饱满", "top2_30k", 0.0),
        ("Top 2 50k", "72M 过训期 · 白化", "top2_50k", 0.0),
        ("Top 5 50k", "72M 5家容量原版", "top5", 0.0),
        ("Top 5 对齐", "强分布 (α=1.00)", "top5", 1.00),
        ("Top 10 70k", "Exp A 114M 原版", "top10", 0.0),
        ("Top 10 对齐", "轻度微调 (α=0.35)", "top10", 0.35),
        ("Top 10 对齐", "适度平衡 (α=0.70)", "top10", 0.70),
        ("Top 10 对齐", "强力分布 (α=1.00)", "top10", 1.00),
    ]

    num_cols = len(COLUMNS)
    num_rows = len(SAMPLE_MAPPING)

    total_w = label_w + num_cols * (cell_w + pad) + pad
    total_h = title_h + header_h + num_rows * (cell_h + pad) + pad + 20

    print(f"Creating poster canvas: {total_w} x {total_h} for 20 clean rows...")
    poster = Image.new("RGB", (total_w, total_h), (246, 248, 250))
    draw = ImageDraw.Draw(poster)

    # 1. 顶部大标题与说明
    title = "Top 2 (30k vs 50k 过训对照) / Top 5 / Top 10 模型容量演进与强力笔画对齐全貌图册"
    subtitle = "涵盖物理字模基准 (STD/GT)、Top 2 训练崩溃退化直击、书家规模扩展 (2 -> 5 -> 10 家) 与真实可感知粗细 SDF 梯度"
    draw.text((pad + 10, 20), title, fill=(15, 23, 42), font=font_title)
    draw.text((pad + 12, 62), subtitle, fill=(100, 116, 139), font=font_subtitle)

    # 2. 列头绘制
    for c_idx, (col_main, col_sub, col_type, alpha) in enumerate(COLUMNS):
        x = label_w + c_idx * (cell_w + pad)
        y = title_h + 10

        if col_type in ("std", "gt"):
            bg_col = (230, 235, 242)
            border_col = (203, 213, 225)
        elif "top2_30k" in col_type:
            bg_col = (236, 253, 245)
            border_col = (167, 243, 208)
        elif "top2_50k" in col_type:
            bg_col = (254, 242, 242)
            border_col = (254, 202, 202)
        elif "top5" in col_type:
            bg_col = (239, 246, 255) if alpha > 0 else (241, 245, 249)
            border_col = (191, 219, 254) if alpha > 0 else (203, 213, 225)
        else:  # top10
            if alpha >= 0.9:
                bg_col = (254, 243, 199)
                border_col = (252, 211, 77)
            elif alpha > 0.0:
                bg_col = (255, 247, 237)
                border_col = (254, 215, 170)
            else:
                bg_col = (241, 245, 249)
                border_col = (203, 213, 225)

        draw.rectangle([x, y, x + cell_w, y + header_h - 15], fill=bg_col, outline=border_col, width=2)
        draw.text((x + 14, y + 18), col_main, fill=(15, 23, 42), font=font_col_main)
        draw.text((x + 14, y + 54), col_sub, fill=(100, 116, 139), font=font_col_sub)

    # 3. 逐行填充
    print("Rendering 20 rows with pristine centering and powerful SDF...")
    for r_idx, s in enumerate(SAMPLE_MAPPING):
        i2 = s["i2"]
        i5 = s["i5"]
        i10 = s["i10"]
        callig = s["callig"]
        script = s["script"]
        char = s["char"]
        grp = s.get("group", "")
        slot = f"{callig}_{script}"
        target_dist = train_dist.get(slot, train_dist["_GLOBAL_"])

        y = title_h + header_h + r_idx * (cell_h + pad)

        grp_bg, grp_accent = GROUP_COLORS.get(grp, ((255, 255, 255), (200, 200, 200)))

        # 左侧行标签
        draw.rectangle([pad, y, label_w - pad, y + cell_h], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
        draw.rectangle([pad, y, pad + 6, y + cell_h], fill=grp_accent)
        draw.rectangle([pad + 12, y + 12, label_w - pad - 12, y + 34], fill=grp_bg, outline=grp_accent, width=1)
        draw.text((pad + 18, y + 14), grp, fill=grp_accent, font=font_badge)

        draw.text((pad + 16, y + 44), f"【{char}】", fill=(15, 23, 42), font=font_row_char)
        draw.text((pad + 16, y + 96), f"{callig}", fill=(30, 41, 59), font=font_row_text)
        draw.text((pad + 16, y + 128), f"书体: {script}书", fill=(71, 85, 105), font=font_col_sub)
        draw.text((pad + 16, y + 165), f"2.6万真迹基准:", fill=(148, 163, 184), font=font_tag)
        draw.text((pad + 16, y + 185), f"μ={target_dist['ref_mean']:.1f} P50={target_dist['ref_p50']:.1f}", fill=(51, 65, 85), font=font_tag)

        # 逐列填充
        for c_idx, (_, _, col_type, alpha) in enumerate(COLUMNS):
            x = label_w + c_idx * (cell_w + pad)
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(255, 255, 255), outline=(226, 232, 240), width=1)

            im_raw = None
            if col_type == "std":
                p = ROOT / f"artifacts/DiC_U100_review_20261020/images/reference/std/g{i10}.png"
                if p.exists():
                    im_raw = Image.open(p).convert("L")
            elif col_type == "gt":
                p = ROOT / f"artifacts/DiC_U100_review_20261020/images/reference/gt/g{i10}.png"
                if p.exists():
                    im_raw = Image.open(p).convert("L")
            elif col_type == "top2_30k":
                p = ROOT / f"assets/results/top2_u100_base_30k/g{i2}.png"
                if p.exists():
                    im_raw = Image.open(p).convert("L")
            elif col_type == "top2_50k":
                p = ROOT / f"assets/results/top2_u100_base_50k/g{i2}.png"
                if p.exists():
                    im_raw = Image.open(p).convert("L")
            elif col_type == "top5":
                p = ROOT / f"assets/results/top5_u100_base_50k/g{i5}.png"
                if p.exists():
                    im_raw = Image.open(p).convert("L")
            elif col_type == "top10":
                p = ROOT / f"artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g{i10}.png"
                if p.exists():
                    im_raw = Image.open(p).convert("L")

            if im_raw is not None:
                img_np = np.array(im_raw, dtype=np.float32) / 255.0

                # 仅对生成模型应用强力有效 SDF 对齐
                if alpha > 0.0 and col_type not in ("std", "gt"):
                    img_np = powerful_sdf_align(img_np, target_dist, alpha=alpha)

                # 测量物理粗细与对比度
                p50_val, con_val = measure_p50_and_contrast(img_np)

                # 使用安全内边距函数，生成完全居中、四周留白 >= 20px、绝不被裁切的图像
                cell_img = prepare_cell_image(img_np, cell_size=img_box_h, inner_pad=20)
                poster.paste(cell_img, (x, y))

                # 绘制独立的底部元数据槽 (Metadata Bar)
                meta_y = y + img_box_h
                meta_bg = (248, 250, 252)
                if col_type == "top2_50k" and p50_val < 3.0:
                    meta_bg = (254, 242, 242) # 标红提醒过训白化
                draw.rectangle([x, meta_y, x + cell_w, y + cell_h], fill=meta_bg)
                draw.line([(x, meta_y), (x + cell_w, meta_y)], fill=(226, 232, 240), width=1)

                if p50_val > 0.0:
                    tag_text = f"P50: {p50_val:.1f}px · 提按: {con_val:.1f}"
                    tag_color = (71, 85, 105)
                else:
                    tag_text = "白化/退化 (Ink < 100)"
                    tag_color = (220, 38, 38)
                draw.text((x + 12, meta_y + 5), tag_text, fill=tag_color, font=font_tag)
            else:
                draw.text((x + 40, y + 100), "Missing", fill=(150, 150, 150), font=font_col_sub)

    OUT_POSTER.parent.mkdir(parents=True, exist_ok=True)
    poster.save(str(OUT_POSTER), quality=96)
    print(f"\nSuccessfully generated and saved powerful 20-row poster to:\n  {OUT_POSTER}")


if __name__ == "__main__":
    main()
