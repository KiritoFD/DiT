#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/render_top5_30_pure_scale_poster.py —
Top 5 三十字简历级全貌图册 (7 列纯神经网络端到端演进版，最右侧加入 DiT L20 架构终局标杆):
  1. 30 组传世名家典范字样 (欧阳询、王羲之、苏轼、赵孟頫、颜真卿 5 家严格均衡，15 训练集 Seen + 15 测试集 Unseen)
  2. 显式标注 [训练集 · Seen] (拟合/重建) 或 [测试集 · Unseen] (盲测/零样本泛化)
  3. 7 大核心纵列 (数据规模、模型容量与 CNN vs Transformer 架构对比):
     - Col 1: STD 参考 (标准字模骨架先验)
     - Col 2: GT 真迹 (传世碑帖法书原稿)
     - Col 3: Top 2 原版 (U-Net 72M 30k 王/苏基线)
     - Col 4: Top 5 原版 (U-Net 72M 50k 5家基线)
     - Col 5: Top 5 Scale (U-Net 114M 40k 纯加深新标杆，彻底剔除解耦，字形稳固)
     - Col 6: Top 10 原版 (U-Net 114M 70k 10家全量标杆)
     - Col 7: DiT L20 (Diffusion Transformer 20L×384 54M 200k 终局标杆)
  4. 自动生成原版 + 2x 高清下采样紧凑版 (极速加载与简历嵌入)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis, skeletonize

ROOT = Path("/root/Workspace/xy/DiT") if Path("/root/Workspace/xy/DiT").exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FONT_PATH = ROOT / "tools/fonts/SimHei.ttf"
if not FONT_PATH.exists():
    FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")

CSV_TOP5_30 = ROOT / "exp-std/csv/eval_top5_30.csv"
CSV_TOP10 = ROOT / "exp-std/csv/eval200_fixed.csv"
CSV_TOP2 = ROOT / "exp-std/csv/eval_top2_100.csv"
TRAIN_DIST_JSON = ROOT / "assets/stroke_width_trainset_distribution.json"

OUT_RAW = ROOT / "assets/posters/poster_top5_30_pure_scale_raw.png"
OUT_2X = ROOT / "assets/posters/poster_top5_30_pure_scale.png"

T10_DIR = ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0070000/eval200fix"
if not T10_DIR.exists():
    T10_DIR = ROOT / "artifacts/DiC_U100_review_20261020/images/originals/decA_70k"

T2_DIR = ROOT / "assets/results/unet_subsets/u100_base_top2_noaug/20261020-143606-Unet-flow/eval_samples_ctrl/step0030000/eval200fix"
if not T2_DIR.exists():
    T2_DIR = ROOT / "assets/results/top2_u100_base_30k"

T5_OLD_DIR = ROOT / "assets/results/unet_subsets/u100_base_top5_noaug/20261020-195826-Unet-flow/eval_samples_ctrl/step0050000/eval200fix"
if not T5_OLD_DIR.exists():
    T5_OLD_DIR = ROOT / "assets/results/top5_u100_base_50k"

T5_SCALE_DIR = ROOT / "assets/results/unet_subsets/top5_u100_deep_sym4_50k/20261021-111658-Unet-flow/eval_samples_ctrl/step0040000/eval200fix"
if not T5_SCALE_DIR.exists():
    T5_SCALE_DIR = ROOT / "assets/results/unet_subsets/top5_u100_deep_sym4_50k/20261021-111658-Unet-flow/eval_samples_ctrl/step0035000/eval200fix"

STD_DIR = ROOT / "assets/results/unet_subsets/top5_u100_deep_sym4_50k/20261021-111658-Unet-flow/eval_samples_ctrl/eval200fix_input_g"
if not STD_DIR.exists():
    STD_DIR = ROOT / "assets/results/unet_subsets/top5_114m_decoupled_50k/20261021-063103-Unet-flow/eval_samples_ctrl/eval200fix_input_g"

DIT_L20_DIR = ROOT / "scratch/v90_20l_gpu_20261017_01/assets/results/v90A_20L_stdstrong/20261017-150613-v90A_20L_stdstrong/eval_samples_ctrl/step0200000/eval200fix"
if not DIT_L20_DIR.exists():
    DIT_L20_DIR = ROOT / "scratch/v90_20l_gpu_20261017_01/assets/results/v90A_20L_stdstrong/20261017-150613-v90A_20L_stdstrong/eval_samples_ctrl/step0190000/eval200fix"

CALLIG_COLORS = {
    "欧阳询": ((239, 246, 255), (37, 99, 235)),    # Blue
    "王羲之": ((240, 253, 250), (13, 148, 136)),   # Teal
    "苏轼":   ((254, 242, 242), (225, 29, 72)),    # Rose
    "赵孟頫": ((250, 245, 255), (147, 51, 234)),   # Purple
    "颜真卿": ((255, 251, 235), (217, 119, 6)),    # Amber
}


def measure_p50_and_contrast(img_np: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
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


def prepare_cell_image(img_np: np.ndarray, cell_size: int = 240, inner_pad: int = 20) -> Image.Image:
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

    if ymin >= 18 and (255 - ymax) >= 18 and xmin >= 18 and (255 - xmax) >= 18:
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


def make_na_card(cell_size: int = 240, font_main=None, font_sub=None) -> Image.Image:
    canvas = Image.new("RGB", (cell_size, cell_size), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([10, 10, cell_size - 10, cell_size - 10], outline=(226, 232, 240), width=1)
    draw.text((cell_size // 2 - 44, cell_size // 2 - 20), "— 未纳入 —", fill=(148, 163, 184), font=font_main)
    draw.text((cell_size // 2 - 62, cell_size // 2 + 8), "(Top 2 仅限王/苏)", fill=(203, 213, 225), font=font_sub)
    return canvas


def main():
    print(f"Loading samples from {CSV_TOP5_30}...")
    with open(CSV_TOP5_30, "r", encoding="utf-8-sig") as f:
        samples = list(csv.DictReader(f))

    print(f"Loading mapping from {CSV_TOP10} & {CSV_TOP2}...")
    r10_csv = list(csv.DictReader(open(CSV_TOP10, encoding="utf-8-sig")))
    r2_csv = list(csv.DictReader(open(CSV_TOP2, encoding="utf-8-sig")))
    img_to_i10 = {r["image_path"]: i for i, r in enumerate(r10_csv)}
    img_to_i2 = {r["image_path"]: i for i, r in enumerate(r2_csv)}

    print(f"Loading training set distribution table from {TRAIN_DIST_JSON}...")
    with open(TRAIN_DIST_JSON, "r", encoding="utf-8") as f:
        train_dist = json.load(f)

    # 字体准备
    font_title = ImageFont.truetype(str(FONT_PATH), 28)
    font_subtitle = ImageFont.truetype(str(FONT_PATH), 17)
    font_col_main = ImageFont.truetype(str(FONT_PATH), 20)
    font_col_sub = ImageFont.truetype(str(FONT_PATH), 13)
    font_row_char = ImageFont.truetype(str(FONT_PATH), 32)
    font_tag = ImageFont.truetype(str(FONT_PATH), 13)
    font_badge = ImageFont.truetype(str(FONT_PATH), 13)
    font_status_badge = ImageFont.truetype(str(FONT_PATH), 13)
    font_na_main = ImageFont.truetype(str(FONT_PATH), 16)
    font_na_sub = ImageFont.truetype(str(FONT_PATH), 13)

    cell_w = 240
    img_box_h = 240
    meta_h = 26
    cell_h = img_box_h + meta_h  # 266px

    pad = 12
    label_w = 220
    header_h = 100
    title_h = 100

    # 7 列纯模型容量演进版（最右侧加入 DiT L20）
    COLUMNS = [
        ("STD 参考", "标准印刷/楷书字模", "std"),
        ("GT 真迹", "传世碑帖法书原稿", "gt"),
        ("Top 2 原版", "72M 30k (王/苏基线)", "top2"),
        ("Top 5 原版", "72M 50k (5家轻量基线)", "top5_old"),
        ("Top 5 Scale", "114M 40k (纯加深新标杆)", "top5_scale"),
        ("Top 10 原版", "114M 70k (10家全量标杆)", "top10"),
        ("DiT L20", "20L×384 54M (200k 终局)", "dit_l20"),
    ]

    num_cols = len(COLUMNS)
    num_rows = len(samples)

    total_w = label_w + num_cols * (cell_w + pad) + pad
    total_h = title_h + header_h + num_rows * (cell_h + pad) + pad + 20

    print(f"Creating poster canvas: {total_w} x {total_h} for 30 rows...")
    poster = Image.new("RGB", (total_w, total_h), (246, 248, 250))
    draw = ImageDraw.Draw(poster)

    # 1. 顶部标题
    title = "Top 5 (五大宗师) 简历级代表作全貌图册 (15 训练集 Seen + 15 测试集 Unseen 严谨对半开)"
    subtitle = "规模演进与架构对照：STD 字模 -> GT 真迹 -> Top 2 (72M) -> Top 5 (72M) -> Top 5 Scale (114M 纯加深) -> Top 10 (114M) -> DiT L20 (54M)"
    draw.text((pad + 10, 20), title, fill=(15, 23, 42), font=font_title)
    draw.text((pad + 12, 62), subtitle, fill=(100, 116, 139), font=font_subtitle)

    # 2. 列头绘制
    for c_idx, (col_main, col_sub, col_type) in enumerate(COLUMNS):
        x = label_w + c_idx * (cell_w + pad)
        y = title_h + 10

        if col_type in ("std", "gt"):
            bg_col = (230, 235, 242)
            border_col = (203, 213, 225)
        elif col_type == "top2":
            bg_col = (236, 253, 245)
            border_col = (167, 243, 208)
        elif col_type == "top5_old":
            bg_col = (241, 245, 249)
            border_col = (203, 213, 225)
        elif col_type == "top5_scale":
            bg_col = (239, 246, 255)
            border_col = (191, 219, 254)
        elif col_type == "top10":
            bg_col = (238, 242, 255)
            border_col = (199, 210, 254)
        else:  # dit_l20
            bg_col = (253, 242, 248)
            border_col = (244, 114, 182)

        draw.rectangle([x, y, x + cell_w, y + header_h - 15], fill=bg_col, outline=border_col, width=2)
        draw.text((x + 14, y + 16), col_main, fill=(15, 23, 42), font=font_col_main)
        draw.text((x + 14, y + 50), col_sub, fill=(100, 116, 139), font=font_col_sub)

    # 3. 逐行填充
    print("Rendering 30 rows with 7 columns...")

    for r_idx, s in enumerate(samples):
        orig_i5 = int(s["orig_eval_idx"])
        callig = s["calligrapher"]
        script = s["script"]
        char = s["character"]
        is_seen = s["is_seen"].lower() == "seen"
        p_img = s["image_path"]

        i10 = img_to_i10.get(p_img, -1)
        i2 = img_to_i2.get(p_img, -1)

        slot = f"{callig}_{script}"
        target_dist = train_dist.get(slot, train_dist["_GLOBAL_"])

        y = title_h + header_h + r_idx * (cell_h + pad)
        grp_bg, grp_accent = CALLIG_COLORS.get(callig, ((255, 255, 255), (100, 100, 100)))

        # 左侧行标签
        draw.rectangle([pad, y, label_w - pad, y + cell_h], fill=(255, 255, 255), outline=(226, 232, 240), width=2)
        draw.rectangle([pad, y, pad + 6, y + cell_h], fill=grp_accent)

        # 书家徽章
        draw.rectangle([pad + 12, y + 10, label_w - pad - 12, y + 32], fill=grp_bg, outline=grp_accent, width=1)
        draw.text((pad + 18, y + 12), f"{callig} · {script}书", fill=grp_accent, font=font_badge)

        # 核心字样
        draw.text((pad + 14, y + 42), f"【{char}】", fill=(15, 23, 42), font=font_row_char)

        # 训练集 / 测试集 醒目标签 (Seen / Unseen)
        badge_y = y + 96
        if is_seen:
            draw.rectangle([pad + 14, badge_y, label_w - pad - 14, badge_y + 24], fill=(236, 253, 245), outline=(16, 185, 129), width=1)
            draw.text((pad + 20, badge_y + 3), "● 训练集 · Seen", fill=(5, 150, 105), font=font_status_badge)
        else:
            draw.rectangle([pad + 14, badge_y, label_w - pad - 14, badge_y + 24], fill=(238, 242, 255), outline=(99, 102, 241), width=1)
            draw.text((pad + 20, badge_y + 3), "◆ 测试集 · Unseen", fill=(67, 56, 202), font=font_status_badge)

        # 基准元数据
        t2_str = f"T2:#{i2:02d}" if i2 != -1 else "T2:—"
        draw.text((pad + 14, y + 134), f"T5:#{orig_i5:02d} / T10:#{i10:03d} / {t2_str}", fill=(100, 116, 139), font=font_col_sub)
        draw.text((pad + 14, y + 160), "真迹基准分布:", fill=(148, 163, 184), font=font_tag)
        draw.text((pad + 14, y + 182), f"μ={target_dist['ref_mean']:.1f} P50={target_dist['ref_p50']:.1f}", fill=(51, 65, 85), font=font_tag)

        # 逐列填充图像
        for c_idx, (_, _, col_type) in enumerate(COLUMNS):
            x = label_w + c_idx * (cell_w + pad)
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(255, 255, 255), outline=(226, 232, 240), width=1)

            # 特殊情况: Top 2 非王/苏书家
            if col_type == "top2" and i2 == -1:
                na_img = make_na_card(cell_size=img_box_h, font_main=font_na_main, font_sub=font_na_sub)
                poster.paste(na_img, (x, y))
                meta_y = y + img_box_h
                draw.rectangle([x, meta_y, x + cell_w, y + cell_h], fill=(241, 245, 249))
                draw.line([(x, meta_y), (x + cell_w, meta_y)], fill=(226, 232, 240), width=1)
                draw.text((x + 12, meta_y + 5), "N/A (仅评测王/苏)", fill=(148, 163, 184), font=font_tag)
                continue

            im_raw = None
            if col_type == "std":
                p1 = STD_DIR / f"g{orig_i5}.png"
                p2 = STD_DIR / f"g{i10}.png"
                p3 = ROOT / f"artifacts/DiC_U100_review_20261020/images/reference/std/g{i10}.png"
                for p in [p1, p2, p3]:
                    if p.exists():
                        im_raw = Image.open(p).convert("L")
                        break
            elif col_type == "gt":
                p1 = T5_SCALE_DIR / f"gt{orig_i5}.png"
                p2 = T5_OLD_DIR / f"gt{orig_i5}.png"
                p3 = ROOT / f"artifacts/DiC_U100_review_20261020/images/reference/gt/g{i10}.png"
                for p in [p1, p2, p3]:
                    if p.exists():
                        im_raw = Image.open(p).convert("L")
                        break
            elif col_type == "top2":
                p1 = T2_DIR / f"g{i2}.png"
                p2 = ROOT / f"assets/results/top2_u100_base_30k/g{i2}.png"
                for p in [p1, p2]:
                    if p.exists():
                        im_raw = Image.open(p).convert("L")
                        break
            elif col_type == "top5_old":
                p1 = T5_OLD_DIR / f"g{orig_i5}.png"
                p2 = ROOT / f"assets/results/top5_u100_base_50k/g{orig_i5}.png"
                for p in [p1, p2]:
                    if p.exists():
                        im_raw = Image.open(p).convert("L")
                        break
            elif col_type == "top5_scale":
                p1 = T5_SCALE_DIR / f"g{orig_i5}.png"
                if p1.exists():
                    im_raw = Image.open(p1).convert("L")
            elif col_type == "top10":
                p1 = T10_DIR / f"g{i10}.png"
                p2 = T10_DIR / f"g{i10:03d}.png"
                p3 = ROOT / f"artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g{i10}.png"
                for p in [p1, p2, p3]:
                    if p.exists():
                        im_raw = Image.open(p).convert("L")
                        break
            elif col_type == "dit_l20":
                p1 = DIT_L20_DIR / f"g{i10}.png"
                if p1.exists():
                    im_raw = Image.open(p1).convert("L")

            if im_raw is not None:
                img_np = np.array(im_raw, dtype=np.float32) / 255.0
                p50_val, con_val = measure_p50_and_contrast(img_np)
                cell_img = prepare_cell_image(img_np, cell_size=img_box_h, inner_pad=20)
                poster.paste(cell_img, (x, y))

                meta_y = y + img_box_h
                meta_bg = (248, 250, 252)
                draw.rectangle([x, meta_y, x + cell_w, y + cell_h], fill=meta_bg)
                draw.line([(x, meta_y), (x + cell_w, meta_y)], fill=(226, 232, 240), width=1)

                if p50_val > 0.0:
                    tag_text = f"P50: {p50_val:.1f}px · 提按: {con_val:.1f}"
                    tag_color = (71, 85, 105)
                else:
                    tag_text = "空白/退化"
                    tag_color = (220, 38, 38)
                draw.text((x + 12, meta_y + 5), tag_text, fill=tag_color, font=font_tag)
            else:
                draw.text((x + 40, y + 100), "Missing", fill=(220, 38, 38), font=font_col_sub)

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    poster.save(str(OUT_RAW), quality=96)
    print(f"Saved raw poster: {OUT_RAW} ({OUT_RAW.stat().st_size / 1024 / 1024:.2f} MB)")

    # 4. 执行 2x 高清下采样
    w_2x = total_w // 2
    h_2x = total_h // 2
    print(f"Downsampling 2x: {total_w}x{total_h} -> {w_2x}x{h_2x}...")
    poster_2x = poster.resize((w_2x, h_2x), Image.Resampling.LANCZOS)
    poster_2x.save(str(OUT_2X), quality=92, optimize=True)
    print(f"Successfully saved 2x downsampled poster: {OUT_2X} ({OUT_2X.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
