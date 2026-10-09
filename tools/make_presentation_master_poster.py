#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/make_presentation_master_poster.py
生成马良 (Callig-DiT) 多阶段演化全景汇报海报 (包含 v10b, v13, v20/v21, v23, v66, v68 阶段)
"""
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

ROOT = "/root/Workspace/xy/DiT"
if os.path.exists(ROOT):
    os.chdir(ROOT)

SEEN_CSV = "assets/eval_seen_v10.csv"
OUT_IMG = "docs/04_experiments/imgs/calligdit_evolution_master_poster.png"

# 8 行定义：Input g -> v10b -> v13 -> v21 -> v23 -> v66 -> v68 -> GT
ROW_CONFIGS = [
    {
        "id": "input_g",
        "title": "【输入条件】标准骨架",
        "sub": "无偏标准印刷体骨架 (std skel)",
        "badge": "纯几何拓扑引导",
        "badge_color": (230, 240, 255),
        "badge_text_color": (30, 80, 160),
        "dir": "assets/results/std_callig_aug/eval_samples_ctrl/seen_input_g",
        "pattern": "g{idx}.png",
        "outline_color": (160, 190, 230)
    },
    {
        "id": "v10b",
        "title": "一阶段: v10b 初代骨架长跑",
        "sub": "去字表纯骨架 / 41位名家全集 / 余弦退火",
        "badge": "Step 390k | Seen 0.7606 | Strict 0.5680",
        "badge_color": (238, 242, 246),
        "badge_text_color": (50, 65, 85),
        "dir": "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/g",
        "pattern": "g{idx}.png",
        "outline_color": (200, 205, 215)
    },
    {
        "id": "v13",
        "title": "二阶段: v13 50k高保真清洗",
        "sub": "黄金50k底库 / 45位大师正交化 / 4ch流匹配",
        "badge": "Step 155k | Seen 0.7580 | Strict 0.5547",
        "badge_color": (238, 242, 246),
        "badge_text_color": (50, 65, 85),
        "dir": "assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl/step0155000/g",
        "pattern": "g{idx}.png",
        "outline_color": (200, 205, 215)
    },
    {
        "id": "v21",
        "title": "三阶段A: v21 初代SkelNet",
        "sub": "引入显式双尺度几何网格形变 / 联合归一化",
        "badge": "Step 75k | 形变网格引入 | 目标特异度 +0.0126",
        "badge_color": (255, 243, 230),
        "badge_text_color": (160, 80, 20),
        "dir": "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl/step0075000/g",
        "pattern": "g{idx}.png",
        "outline_color": (235, 195, 160)
    },
    {
        "id": "v23",
        "title": "三阶段B: v23 独立归一化",
        "sub": "修复形变通道均值漂移 / 分通道独立LN",
        "badge": "Step 75k | 接口修复 | 目标特异度 +0.0164",
        "badge_color": (255, 243, 230),
        "badge_text_color": (160, 80, 20),
        "dir": "archive_experiments/_archive_historical/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl/step0075000/g",
        "pattern": "g{idx}.png",
        "outline_color": (235, 195, 160)
    },
    {
        "id": "v66",
        "title": "四阶段: v66 风格层级路由",
        "sub": "多尺度AdaLN逐层注入(Block 2,4,5,6) / 路由解耦",
        "badge": "Step 150k | 层次风格调制 | 笔画质感跃升",
        "badge_color": (235, 248, 238),
        "badge_text_color": (30, 115, 60),
        "dir": "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/g",
        "pattern": "g{idx}.png",
        "outline_color": (175, 220, 190)
    },
    {
        "id": "v68",
        "title": "五阶段: v68 3x对称增广+C2OT",
        "sub": "77.8k全量增广底库 / C2OT流匹配 / REPA对齐",
        "badge": "Step 200k | 旗舰综合基准 | 苍劲墨韵与神态形变",
        "badge_color": (230, 245, 255),
        "badge_text_color": (20, 95, 170),
        "dir": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/g",
        "pattern": "g{idx}.png",
        "outline_color": (150, 195, 235)
    },
    {
        "id": "gt",
        "title": "【真值基准】古代名家真迹",
        "sub": "历代传世碑帖与墨迹原拓真实真值",
        "badge": "Ground Truth 历史真迹基准",
        "badge_color": (255, 238, 238),
        "badge_text_color": (160, 35, 35),
        "dir": "assets/results/v13_base_50k/20260917-211905-v13-base-50k/eval_samples_ctrl/step0155000/g",
        "pattern": "gt{idx}.png",
        "outline_color": (235, 175, 175)
    }
]

def get_font(size, bold=False):
    candidates = [
        "tools/fonts/simkai.ttf",
        "tools/fonts/SimHei.ttf",
        "/root/Workspace/xy/DiT/tools/fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf"
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()

def main():
    print("=== 开始制作 Callig-DiT 跨代演进全景汇报海报 ===")
    df_seen = pd.read_csv(SEEN_CSV)
    n_cols = len(df_seen)  # 10 columns
    print(f"载入评测样本：{n_cols} 个字符 (全部基准)")

    cell_size = 200
    row_header_w = 340
    top_header_h = 145
    col_header_h = 60
    padding = 16
    gap = 8

    n_rows = len(ROW_CONFIGS)  # 8 rows

    canvas_w = row_header_w + n_cols * cell_size + (n_cols - 1) * gap + padding * 2
    canvas_h = top_header_h + col_header_h + n_rows * cell_size + (n_rows - 1) * gap + padding * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 249, 252))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(30, bold=True)
    sub_title_font = get_font(15)
    tag_font = get_font(15, bold=True)
    row_title_font = get_font(18, bold=True)
    row_sub_font = get_font(12)
    row_badge_font = get_font(12, bold=True)
    col_char_font = get_font(20, bold=True)
    col_desc_font = get_font(13)

    # 1. 顶部 Header
    draw.rectangle([0, 0, canvas_w, top_header_h], fill=(24, 28, 38))
    draw.text((padding + 12, 22), "马良 (Callig-DiT) 跨代演进与多阶段全景对比汇报海报", font=title_font, fill=(255, 255, 255))
    subtitle = "全景演进脉络：初代骨架(v10b) → 50k高保真清洗(v13) → 显式SkelNet形变(v21/v23) → 风格层级路由(v66) → 3x对称增广+C2OT旗舰(v68)"
    draw.text((padding + 14, 70), subtitle, font=sub_title_font, fill=(180, 195, 215))
    summary_tag = "核心演进结论：从早期字表强过拟合/粗糙笔触，经由SkelNet解耦形变与层级路由，最终在v68实现名家特异形变与真实碑帖墨韵的完美融合！"
    draw.text((padding + 14, 98), summary_tag, font=tag_font, fill=(255, 212, 100))

    # 2. 列标题 (10 个字符及书家信息)
    y_col_hdr = top_header_h + padding
    for c_idx, r in df_seen.iterrows():
        x = row_header_w + padding + c_idx * (cell_size + gap)
        # 背景底卡片
        draw.rectangle([x, y_col_hdr, x + cell_size, y_col_hdr + col_header_h - 4], fill=(234, 238, 245), outline=(210, 216, 228), width=1)
        tag_char = f"{r['character']}"
        tag_desc = f"{r['calligrapher']} · {r['script']}书"
        draw.text((x + 14, y_col_hdr + 8), tag_char, font=col_char_font, fill=(20, 25, 35))
        draw.text((x + 48, y_col_hdr + 14), tag_desc, font=col_desc_font, fill=(80, 95, 115))
        draw.text((x + 14, y_col_hdr + 36), f"Sample #{c_idx}", font=get_font(11), fill=(130, 140, 155))

    # 3. 逐行绘制
    y_start = y_col_hdr + col_header_h
    for r_idx, r_cfg in enumerate(ROW_CONFIGS):
        y = y_start + r_idx * (cell_size + gap)

        # 3.1 绘制行左侧说明栏
        draw.rectangle([padding, y, row_header_w - 12, y + cell_size], fill=(255, 255, 255), outline=r_cfg["outline_color"], width=2)
        draw.text((padding + 16, y + 20), r_cfg["title"], font=row_title_font, fill=(20, 30, 45))
        draw.text((padding + 16, y + 55), r_cfg["sub"], font=row_sub_font, fill=(100, 110, 125))

        # 指标 Badge 徽章
        badge_y1 = y + 120
        badge_y2 = y + 165
        draw.rectangle([padding + 14, badge_y1, row_header_w - 26, badge_y2], fill=r_cfg["badge_color"], outline=r_cfg["outline_color"], width=1)
        draw.text((padding + 22, badge_y1 + 10), r_cfg["badge"], font=row_badge_font, fill=r_cfg["badge_text_color"])

        # 3.2 绘制 10 列对应的图片
        for c_idx in range(n_cols):
            x = row_header_w + padding + c_idx * (cell_size + gap)
            file_name = r_cfg["pattern"].format(idx=c_idx)
            img_path = os.path.join(r_cfg["dir"], file_name)

            if os.path.exists(img_path):
                im = Image.open(img_path).convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            else:
                im = Image.new("RGB", (cell_size, cell_size), (242, 242, 245))
                dim = ImageDraw.Draw(im)
                dim.text((cell_size // 4, cell_size // 2 - 10), "NOT FOUND", fill=(180, 80, 80))

            canvas.paste(im, (x, y))
            # 边框
            border_col = (200, 40, 40) if r_cfg["id"] == "gt" else ((40, 130, 220) if r_cfg["id"] == "v68" else (220, 225, 232))
            border_w = 2 if r_cfg["id"] in ("v68", "gt") else 1
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=border_col, width=border_w)

    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    canvas.save(OUT_IMG, optimize=True, quality=95)
    print(f"🎉 汇报全景 Master Poster 成功生成至: {OUT_IMG}")
    sz_mb = os.path.getsize(OUT_IMG) / 1024 / 1024
    print(f"   尺寸: {canvas_w}x{canvas_h} px, 大小: {sz_mb:.2f} MB")

if __name__ == "__main__":
    main()
