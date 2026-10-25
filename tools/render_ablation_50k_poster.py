#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/render_ablation_50k_poster.py — 生成横跨 Base、DiC、Arm 1、Arm 2、Exp A (114M) 与 Exp B (在训) 的多模型对比海报
"""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/Workspace/xy/DiT")
FONT_PATH = ROOT / "tools/fonts/SimHei.ttf"
CSV_PATH = ROOT / "exp-std/csv/eval200_fixed.csv"
OUT_PATH = ROOT / "scratch/unet_decoupled_114m/poster_phase_ablation_114m.png"

COLUMNS = [
    ("STD 参考", None, "std"),
    ("GT 真迹", None, "gt"),
    ("Base 55k\n(72M 原版)", ROOT / "exp_img/unet/u100_base_55k", "gen"),
    ("DiC 50k\n(75M 门控+GELU)", ROOT / "exp_img/unet/u100_dic_b768_50k", "gen"),
    ("Arm 2 50k\n(72M 限制层FiLM)", ROOT / "assets/results/unet_ablation_50k/arm2_writer_film_dec_b768/20261019-105029-Unet-flow/eval_samples_ctrl/step0050000/eval200fix", "gen"),
    ("Exp A 50k\n(114M 条件解耦)", ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_50k/20261019-191851-Unet-flow/eval_samples_ctrl/step0050000/eval200fix", "gen"),
    ("Exp A 70k\n(114M 续训结案)", ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0070000/eval200fix", "gen"),
    ("Exp B 5k\n(114M 在训STD残差)", ROOT / "assets/results/unet_decoupled_114m/armB_writer_decoupled_std_70k/20261020-102201-Unet-flow/eval_samples_ctrl/step0005000/eval200fix", "gen"),
]

INDICES = [88, 103, 46, 133, 155, 0, 75, 80, 121, 172]


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    
    font_large = ImageFont.truetype(str(FONT_PATH), 26)
    font_mid = ImageFont.truetype(str(FONT_PATH), 20)
    font_small = ImageFont.truetype(str(FONT_PATH), 16)
    
    cell_w, cell_h = 240, 240
    pad = 12
    label_w = 210
    header_h = 100
    title_h = 70
    
    num_cols = len(COLUMNS)
    num_rows = len(INDICES)
    
    total_w = label_w + num_cols * (cell_w + pad) + pad
    total_h = title_h + header_h + num_rows * (cell_h + pad) + pad
    
    poster = Image.new("RGB", (total_w, total_h), (248, 249, 250))
    draw = ImageDraw.Draw(poster)
    
    # Title
    title = "U-Net 视觉质量与条件解耦阶段性对比海报 (Base vs DiC vs Arm 1/2 vs 114M Decoupled)"
    draw.text((pad, 18), title, fill=(20, 20, 20), font=font_large)
    
    # Headers
    for c_idx, (col_title, _, _) in enumerate(COLUMNS):
        x = label_w + c_idx * (cell_w + pad)
        y = title_h + 10
        # Draw column header background
        draw.rectangle([x, y, x + cell_w, y + header_h - 15], fill=(235, 238, 242), outline=(210, 215, 222))
        lines = col_title.split("\n")
        if len(lines) == 1:
            draw.text((x + 12, y + 24), lines[0], fill=(30, 41, 59), font=font_mid)
        else:
            draw.text((x + 12, y + 14), lines[0], fill=(15, 23, 42), font=font_mid)
            draw.text((x + 12, y + 44), lines[1], fill=(100, 116, 139), font=font_small)
            
    # Rows
    for r_idx, idx in enumerate(INDICES):
        r_meta = rows[idx]
        ch = r_meta["character"]
        callig = r_meta["calligrapher"]
        script = r_meta["script"]
        
        y = title_h + header_h + r_idx * (cell_h + pad)
        
        # Row label background
        draw.rectangle([pad, y, label_w - pad, y + cell_h], fill=(255, 255, 255), outline=(226, 232, 240))
        draw.text((pad + 16, y + 30), f"#{idx:03d} 【{ch}】", fill=(15, 23, 42), font=font_large)
        draw.text((pad + 16, y + 80), f"{callig}", fill=(51, 65, 85), font=font_mid)
        draw.text((pad + 16, y + 115), f"书体: {script}书", fill=(100, 116, 139), font=font_small)
        
        for c_idx, (_, col_root, col_type) in enumerate(COLUMNS):
            x = label_w + c_idx * (cell_w + pad)
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(255, 255, 255), outline=(226, 232, 240))
            
            img_path = None
            if col_type == "std":
                img_path = ROOT / r_meta["std_path"]
            elif col_type == "gt":
                img_path = ROOT / r_meta["image_path"]
            else:
                img_path = col_root / f"g{idx}.png"
                
            if img_path and img_path.exists():
                try:
                    im = Image.open(img_path).convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                    poster.paste(im, (x, y))
                except Exception as e:
                    draw.text((x + 10, y + 100), f"Err: {e}", fill=(200, 0, 0), font=font_small)
            else:
                draw.text((x + 40, y + 100), "Missing", fill=(150, 150, 150), font=font_small)
                
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    poster.save(str(OUT_PATH), quality=95)
    print(f"Poster successfully saved to {OUT_PATH} (size: {poster.size})")


if __name__ == "__main__":
    main()
