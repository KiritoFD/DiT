#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/make_cross_generation_poster.py — 在 4090 服务器上生成同 Step 75,000 跨代对比海报"""
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

EVAL_CSV = "assets/eval_v13_strict_fixed.csv"
OUT_IMG = "docs/04_experiments/imgs/step75k_cross_generation_strict_poster.png"

# 4 代模型在 step 75,000 的样本目录
MODELS = [
    {
        "name": "一代基线 Base (v13)",
        "sub": "无SkelNet / 纯50k底库",
        "ssim": "SSIM 0.5450 | MSE 0.973",
        "dir": "assets/results/v13_base_50k/eval_samples_ctrl/step0075000/strict"
    },
    {
        "name": "二代 SkelNet (v21)",
        "sub": "引入几何变形网格 / 50k",
        "ssim": "SSIM 0.5418 | tgt +0.0126",
        "dir": "assets/results/v21_skelnet_200k/eval_samples_ctrl/step0075000/strict"
    },
    {
        "name": "三代 加15k字模 (std_aug)",
        "sub": "Co-Slot 轮转补齐 / 66k",
        "ssim": "SSIM 0.5544 | tgt +0.0161",
        "dir": "assets/results/std_callig_aug/eval_samples_ctrl/step0075000/strict"
    },
    {
        "name": "四代 独立归一化 (v23)",
        "sub": "split LN 接口修复 / 66k",
        "ssim": "SSIM 0.5537 | tgt +0.0164",
        "dir": "assets/results/v23_splitnorm/eval_samples_ctrl/step0075000/strict"
    }
]

# 骨架与 GT 目录
INPUT_G_DIR = "assets/results/v23_splitnorm/eval_samples_ctrl/strict_input_g"
# 挑选 16 个兼具楷行隶、各大书法名家的经典生僻字样本
PICK_INDICES = [0, 2, 4, 7, 8, 9, 10, 15, 16, 18, 19, 21, 23, 27, 29, 35]

def get_font(size, bold=False):
    # 尝试加载支持中文的字体
    candidates = [
        "tools/fonts/simkai.ttf",
        "tools/fonts/SimHei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()

def main():
    df_eval = pd.read_csv(EVAL_CSV)
    print(f"=== 开始制作 Step 75,000 跨代对比海报 (样本数: {len(PICK_INDICES)}) ===")

    cell_size = 180
    row_header_w = 260
    top_header_h = 130
    col_header_h = 56
    padding = 16
    gap = 6

    n_cols = len(PICK_INDICES)
    n_rows = 1 + len(MODELS) + 1  # Input_g + 4 Models + GT = 6 rows

    canvas_w = row_header_w + n_cols * cell_size + (n_cols - 1) * gap + padding * 2
    canvas_h = top_header_h + col_header_h + n_rows * cell_size + (n_rows - 1) * gap + padding * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(30, bold=True)
    sub_title_font = get_font(16)
    row_title_font = get_font(18, bold=True)
    row_sub_font = get_font(13)
    col_font = get_font(15, bold=True)
    col_sub_font = get_font(13)

    # 1. 绘制海报顶部标题
    draw.rectangle([0, 0, canvas_w, top_header_h], fill=(30, 36, 48))
    draw.text((padding + 10, 22), "马良 (Callig-DiT) 跨代模型同步数对比海报 (Step 75,000 Strict 零样本泛化)", font=title_font, fill=(255, 255, 255))
    subtitle = "评测协议：统一无损内存内评测 (in_mem_eval, 50步 Heun) | 对照对象：一代基线 vs SkelNet变形网络 vs 第一次加数据(15.5k字库) vs 接口独立归一化(v23)"
    draw.text((padding + 12, 68), subtitle, font=sub_title_font, fill=(185, 195, 210))
    summary_tag = "核心结论：第75k步时，字模补丁与接口修复全面反超早期 Base 与初代 SkelNet；行气连带与笔画封闭性大幅提升！"
    draw.text((padding + 12, 94), summary_tag, font=sub_title_font, fill=(255, 215, 110))

    # 2. 绘制列标题 (每个汉字/书家信息)
    y_col_hdr = top_header_h + padding
    for c_idx, sample_idx in enumerate(PICK_INDICES):
        r = df_eval.iloc[sample_idx]
        x = row_header_w + padding + c_idx * (cell_size + gap)
        
        # 背景底色
        draw.rectangle([x, y_col_hdr, x + cell_size, y_col_hdr + col_header_h - 4], fill=(235, 238, 243), outline=(210, 215, 225))
        tag1 = f"{r['character']} · {r['script']}"
        tag2 = f"{r['calligrapher']}"
        
        draw.text((x + 12, y_col_hdr + 8), tag1, font=col_font, fill=(20, 25, 35))
        draw.text((x + 12, y_col_hdr + 30), tag2, font=col_sub_font, fill=(90, 100, 115))

    # 3. 准备各行配置 (Row 0: Input g, Row 1..4: Models, Row 5: GT)
    row_configs = []
    # Row 0: Input g
    row_configs.append({
        "title": "输入标准字骨架 (g)",
        "sub": "无偏印刷体拓扑引导",
        "metric": "输入条件",
        "bg_color": (240, 243, 248),
        "get_img": lambda s_idx: os.path.join(INPUT_G_DIR, f"g{s_idx}.png")
    })
    # Row 1..4: Models
    for m in MODELS:
        row_configs.append({
            "title": m["name"],
            "sub": m["sub"],
            "metric": m["ssim"],
            "bg_color": (255, 255, 255),
            "get_img": lambda s_idx, d=m["dir"]: os.path.join(d, f"g{s_idx}.png")
        })
    # Row 5: GT
    row_configs.append({
        "title": "真实名家真迹 (GT)",
        "sub": "古代碑帖真值标准",
        "metric": "Ground Truth",
        "bg_color": (245, 240, 235),
        "get_img": lambda s_idx: os.path.join(MODELS[-1]["dir"], f"gt{s_idx}.png")
    })

    # 4. 逐行绘制
    y_start = y_col_hdr + col_header_h
    for r_idx, r_cfg in enumerate(row_configs):
        y = y_start + r_idx * (cell_size + gap)
        
        # 4.1 绘制行头 (左侧说明栏)
        draw.rectangle([padding, y, row_header_w - 10, y + cell_size], fill=r_cfg["bg_color"], outline=(215, 220, 230), width=1)
        draw.text((padding + 14, y + 25), r_cfg["title"], font=row_title_font, fill=(20, 30, 45))
        draw.text((padding + 14, y + 60), r_cfg["sub"], font=row_sub_font, fill=(100, 110, 125))
        
        # 指标徽章底色
        badge_color = (230, 245, 235) if "0.55" in r_cfg["metric"] else ((255, 240, 230) if "GT" in r_cfg["metric"] else (240, 242, 246))
        draw.rectangle([padding + 14, y + 105, row_header_w - 24, y + 145], fill=badge_color, outline=(200, 215, 205))
        draw.text((padding + 22, y + 115), r_cfg["metric"], font=row_sub_font, fill=(35, 100, 60) if "0.55" in r_cfg["metric"] else (60, 70, 85))

        # 4.2 绘制该行各个单元格图片
        for c_idx, sample_idx in enumerate(PICK_INDICES):
            x = row_header_w + padding + c_idx * (cell_size + gap)
            img_p = r_cfg["get_img"](sample_idx)

            if os.path.exists(img_p):
                im = Image.open(img_p).convert("RGB").resize((cell_size, cell_size), Image.LANCZOS)
            else:
                im = Image.new("RGB", (cell_size, cell_size), (240, 240, 240))
                d_im = ImageDraw.Draw(im)
                d_im.text((cell_size // 4, cell_size // 2 - 10), "N/A", fill=(160, 160, 160))

            canvas.paste(im, (x, y))
            # 描边框
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(220, 224, 232), width=1)

    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    canvas.save(OUT_IMG, optimize=True)
    print(f"🎉 跨代同步数对比海报已成功生成并落盘至: {OUT_IMG}")
    sz_mb = os.path.getsize(OUT_IMG) / 1024 / 1024
    print(f"   尺寸: {canvas_w}x{canvas_h} px, 大小: {sz_mb:.2f} MB")

if __name__ == "__main__":
    main()
