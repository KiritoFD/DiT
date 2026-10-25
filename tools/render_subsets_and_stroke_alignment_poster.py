#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/render_subsets_and_stroke_alignment_poster.py —
综合大型对比海报 (20 行完整代表字样):
  1. 横向模型容量与书家规模演进: Top 2 (50k) -> Top 5 (50k) -> Top 10 (Exp A 70k 114M)
  2. 纵向多强度笔画粗细空间变异 SDF 对齐:
     - Top 2 原版 vs 强对齐 (α = 1.0)
     - Top 5 原版 vs 强对齐 (α = 1.0)
     - Top 10 原版 vs 轻度 (α = 0.35) vs 适中 (α = 0.70) vs 强分布 (α = 1.00)
  3. 左侧标准字模 (STD) 与古代真迹拓片 (GT) 作为黄金物理锚点
  4. 底部半透明标签实时标注文物级笔宽 P50 与提按比 (Contrast)
  5. 20 行样本分四大风格组严格平衡:
     - 王羲之 楷书 (5字: 旨, 尚, 位, 兩, 恵)
     - 王羲之 行书 (5字: 好, 致, 夜, 羲, 将)
     - 苏轼 楷书 (5字: 靑, 衆, 照, 皇, 鵞)
     - 苏轼 行书 (5字: 危, 稜, 渚, 冒, 桑)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import medial_axis, skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FONT_PATH = ROOT / "tools/fonts/SimHei.ttf"
if not FONT_PATH.exists():
    FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")

TRAIN_DIST_JSON = ROOT / "assets/stroke_width_trainset_distribution.json"
OUT_POSTER = ROOT / "assets/posters/poster_subsets_and_stroke_alignment.png"

# 20 个完整代表性样本 (四大风格组各 5 字严格平衡)
SAMPLE_MAPPING = [
    # 组 1: 王羲之 楷书 (清健俊朗，中锋微敛)
    {"i2": 0,  "i5": 18, "i10": 80,  "callig": "王羲之", "script": "楷", "char": "旨", "group": "王羲之 · 楷书"},
    {"i2": 1,  "i5": 19, "i10": 81,  "callig": "王羲之", "script": "楷", "char": "尚", "group": "王羲之 · 楷书"},
    {"i2": 2,  "i5": 20, "i10": 82,  "callig": "王羲之", "script": "楷", "char": "位", "group": "王羲之 · 楷书"},
    {"i2": 4,  "i5": 22, "i10": 84,  "callig": "王羲之", "script": "楷", "char": "兩", "group": "王羲之 · 楷书"},
    {"i2": 6,  "i5": 24, "i10": 86,  "callig": "王羲之", "script": "楷", "char": "恵", "group": "王羲之 · 楷书"},

    # 组 2: 王羲之 行书 (兰亭逸气，牵丝引带)
    {"i2": 25, "i5": 26, "i10": 88,  "callig": "王羲之", "script": "行", "char": "好", "group": "王羲之 · 行书"},
    {"i2": 26, "i5": 27, "i10": 89,  "callig": "王羲之", "script": "行", "char": "致", "group": "王羲之 · 行书"},
    {"i2": 27, "i5": 28, "i10": 90,  "callig": "王羲之", "script": "行", "char": "夜", "group": "王羲之 · 行书"},
    {"i2": 28, "i5": 29, "i10": 91,  "callig": "王羲之", "script": "行", "char": "羲", "group": "王羲之 · 行书"},
    {"i2": 33, "i5": 34, "i10": 96,  "callig": "王羲之", "script": "行", "char": "将", "group": "王羲之 · 行书"},

    # 组 3: 苏轼 楷书 (重墨宽博，端庄厚重)
    {"i2": 50, "i5": 35, "i10": 111, "callig": "苏轼",   "script": "楷", "char": "靑", "group": "苏轼 · 楷书"},
    {"i2": 51, "i5": 36, "i10": 112, "callig": "苏轼",   "script": "楷", "char": "衆", "group": "苏轼 · 楷书"},
    {"i2": 53, "i5": 38, "i10": 114, "callig": "苏轼",   "script": "楷", "char": "照", "group": "苏轼 · 楷书"},
    {"i2": 56, "i5": 41, "i10": 117, "callig": "苏轼",   "script": "楷", "char": "皇", "group": "苏轼 · 楷书"},
    {"i2": 57, "i5": 42, "i10": 118, "callig": "苏轼",   "script": "楷", "char": "鵞", "group": "苏轼 · 楷书"},

    # 组 4: 苏轼 行书 (绵中裹铁，沉着痛快)
    {"i2": 75, "i5": 44, "i10": 120, "callig": "苏轼",   "script": "行", "char": "危", "group": "苏轼 · 行书"},
    {"i2": 77, "i5": 46, "i10": 122, "callig": "苏轼",   "script": "行", "char": "稜", "group": "苏轼 · 行书"},
    {"i2": 78, "i5": 47, "i10": 123, "callig": "苏轼",   "script": "行", "char": "渚", "group": "苏轼 · 行书"},
    {"i2": 79, "i5": 48, "i10": 124, "callig": "苏轼",   "script": "行", "char": "冒", "group": "苏轼 · 行书"},
    {"i2": 82, "i5": 51, "i10": 127, "callig": "苏轼",   "script": "行", "char": "桑", "group": "苏轼 · 行书"},
]

GROUP_COLORS = {
    "王羲之 · 楷书": ((238, 242, 255), (99, 102, 241)),   # Indigo
    "王羲之 · 行书": ((240, 253, 250), (20, 184, 166)),   # Teal
    "苏轼 · 楷书":   ((250, 245, 255), (168, 85, 247)),  # Purple
    "苏轼 · 行书":   ((255, 251, 235), (245, 158, 11)),   # Amber
}


def measure_p50_and_contrast(img_np: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """计算图像的 P50 和提按比。"""
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


def align_with_alpha(
    img_np: np.ndarray,
    target_dist: dict,
    alpha: float = 1.0,
    threshold: float = 0.5,
    max_shift: float = 2.5
) -> np.ndarray:
    """按指定 alpha 强度进行空间变异 SDF 对齐。"""
    if abs(alpha) < 1e-4:
        return img_np

    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 25:
        return img_np

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

    delta_skel = (target_widths - skel_widths) / 2.0
    delta_skel = np.clip(delta_skel, -max_shift, max_shift)

    _, (nearest_r, nearest_c) = distance_transform_edt(~skel, return_indices=True)
    delta_grid = np.zeros_like(img_np, dtype=np.float32)
    delta_grid[skel] = delta_skel.astype(np.float32)
    delta_map = delta_grid[nearest_r, nearest_c]
    delta_map = gaussian_filter(delta_map, sigma=1.2)

    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    phi_prime = phi + alpha * delta_map
    adjusted_ink = np.clip(phi_prime + 0.5, 0.0, 1.0)
    return (1.0 - adjusted_ink).astype(np.float32)


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

    cell_w, cell_h = 240, 240
    pad = 12
    label_w = 210
    header_h = 110
    title_h = 100

    COLUMNS = [
        ("STD 参考", "标准行/楷字模", "std", 0.0),
        ("GT 真迹", "古代传世碑帖", "gt", 0.0),
        ("Top 2 50k", "原版无增强", "top2", 0.0),
        ("Top 2 对齐", "强分布 (α=1.0)", "top2", 1.0),
        ("Top 5 50k", "原版无增强", "top5", 0.0),
        ("Top 5 对齐", "强分布 (α=1.0)", "top5", 1.0),
        ("Top 10 70k", "Exp A 114M 原版", "top10", 0.0),
        ("Top 10 对齐", "轻度微调 (α=0.35)", "top10", 0.35),
        ("Top 10 对齐", "适度平衡 (α=0.70)", "top10", 0.70),
        ("Top 10 对齐", "强分布 (α=1.00)", "top10", 1.00),
    ]

    num_cols = len(COLUMNS)
    num_rows = len(SAMPLE_MAPPING)

    total_w = label_w + num_cols * (cell_w + pad) + pad
    total_h = title_h + header_h + num_rows * (cell_h + pad) + pad + 20

    print(f"Creating poster canvas: {total_w} x {total_h} for {num_rows} rows...")
    poster = Image.new("RGB", (total_w, total_h), (246, 248, 250))
    draw = ImageDraw.Draw(poster)

    # 1. 顶部大标题与说明
    title = "Top 2 / Top 5 / Top 10 模型容量演进与不同强度笔画粗细空间变异 SDF 对齐全貌海报 (20 行对比图册)"
    subtitle = "涵盖纯正真实行书/楷书字模 (STD)、真迹锚点 (GT)、书家规模扩展 (2 -> 5 -> 10 家) 与多阶 SDF 均值/方差/分位数空间对齐梯度"
    draw.text((pad + 10, 20), title, fill=(15, 23, 42), font=font_title)
    draw.text((pad + 12, 62), subtitle, fill=(100, 116, 139), font=font_subtitle)

    # 2. 列头绘制
    for c_idx, (col_main, col_sub, col_type, alpha) in enumerate(COLUMNS):
        x = label_w + c_idx * (cell_w + pad)
        y = title_h + 10

        if col_type in ("std", "gt"):
            bg_col = (230, 235, 242)
            border_col = (203, 213, 225)
        elif "top2" in col_type:
            bg_col = (236, 253, 245) if alpha > 0 else (241, 245, 249)
            border_col = (167, 243, 208) if alpha > 0 else (203, 213, 225)
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

    # 3. 逐行填充样本与实时粗细指标计算
    print("Rendering 20 rows and applying progressive SDF alignment...")
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

        # 组别颜色条与背景
        grp_bg, grp_accent = GROUP_COLORS.get(grp, ((255, 255, 255), (200, 200, 200)))

        # 左侧行标签
        draw.rectangle([pad, y, label_w - pad, y + cell_h], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
        # 组别侧边指示色条
        draw.rectangle([pad, y, pad + 6, y + cell_h], fill=grp_accent)

        # 组别微徽章
        draw.rectangle([pad + 12, y + 10, label_w - pad - 12, y + 30], fill=grp_bg, outline=grp_accent, width=1)
        draw.text((pad + 18, y + 12), grp, fill=grp_accent, font=font_badge)

        draw.text((pad + 16, y + 36), f"【{char}】", fill=(15, 23, 42), font=font_row_char)
        draw.text((pad + 16, y + 88), f"{callig}", fill=(30, 41, 59), font=font_row_text)
        draw.text((pad + 16, y + 120), f"书体: {script}书", fill=(71, 85, 105), font=font_col_sub)
        draw.text((pad + 16, y + 152), f"2.6万真迹基准:", fill=(148, 163, 184), font=font_tag)
        draw.text((pad + 16, y + 172), f"μ={target_dist['ref_mean']:.1f} P50={target_dist['ref_p50']:.1f}", fill=(51, 65, 85), font=font_tag)

        # 逐列图像载入与处理
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
            elif col_type == "top2":
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

                # 若需对齐且 alpha > 0
                if alpha > 0.0:
                    img_np = align_with_alpha(img_np, target_dist, alpha=alpha)

                # 测量显示粗细与提按比
                p50_val, con_val = measure_p50_and_contrast(img_np)

                # 缩放并贴图
                final_im = Image.fromarray((img_np * 255.0).astype(np.uint8)).convert("RGB")
                final_im = final_im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                poster.paste(final_im, (x, y))

                # 绘制底部数据胶囊标签 (Tag)
                tag_text = f"P50: {p50_val:.1f}px | 提按: {con_val:.1f}"
                tag_h = 22
                draw.rectangle([x, y + cell_h - tag_h, x + cell_w, y + cell_h], fill=(245, 247, 250, 220))
                draw.line([(x, y + cell_h - tag_h), (x + cell_w, y + cell_h - tag_h)], fill=(226, 232, 240), width=1)
                draw.text((x + 8, y + cell_h - tag_h + 3), tag_text, fill=(71, 85, 105), font=font_tag)
            else:
                draw.text((x + 40, y + 100), "Missing", fill=(150, 150, 150), font=font_col_sub)

    OUT_POSTER.parent.mkdir(parents=True, exist_ok=True)
    poster.save(str(OUT_POSTER), quality=96)
    print(f"\nSuccessfully generated and saved 20-row combined poster to:\n  {OUT_POSTER}")


if __name__ == "__main__":
    main()
