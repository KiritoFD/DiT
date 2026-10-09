#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/render_poster_from_tree.py
基于本地目录树 (assets/poster_tree) 自动扫描并拼接全景演进汇报 Poster。

特点与增量更新逻辑：
1. 完全依赖目录结构驱动，不硬编码阶段；
2. 增量更新新阶段时，只需在 tree_dir 下建立例如 `07_v70_aug_stdskel/`，放入 `meta.json` 及 `00.png`..`09.png` 即可自动渲染；
3. 若某阶段图片未完全就绪，会自动绘制优雅的「待生成 / In Progress」卡片，防止脚本报错中断；
4. 针对不同 badge_type (input / default / warning / success / accent / gt) 自动赋予统一的专业配色体系。
"""
import os
import sys
import argparse
import json
from PIL import Image, ImageDraw, ImageFont

# 兼容 Windows 终端编码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 预设高对比度汇报配色方案
BADGE_PALETTE = {
    "input": {
        "box_bg": (230, 240, 255),
        "text": (25, 75, 160),
        "border": (170, 200, 240)
    },
    "default": {
        "box_bg": (240, 243, 248),
        "text": (50, 65, 85),
        "border": (210, 215, 225)
    },
    "warning": {
        "box_bg": (255, 243, 230),
        "text": (165, 80, 20),
        "border": (235, 195, 160)
    },
    "success": {
        "box_bg": (235, 248, 238),
        "text": (28, 120, 60),
        "border": (175, 220, 190)
    },
    "accent": {
        "box_bg": (228, 242, 255),
        "text": (18, 90, 175),
        "border": (145, 190, 235)
    },
    "gt": {
        "box_bg": (255, 238, 238),
        "text": (170, 35, 35),
        "border": (240, 175, 175)
    }
}


def find_font(size: int, bold: bool = False):
    """跨平台寻找最优雅的中文字体"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc" if not bold else "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "tools/fonts/SimHei.ttf",
        "/root/Workspace/xy/DiT/tools/fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_columns(tree_dir: str):
    """读取列定义或自动从样本中推断"""
    meta_path = os.path.join(tree_dir, "meta_columns.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 默认兜底 10 列
    return [{"idx": i, "char": f"字{i}", "callig": "名家", "script": "楷"} for i in range(10)]


def load_stages(tree_dir: str):
    """扫描 tree_dir 下的所有子目录作为各个对比阶段"""
    stages = []
    subdirs = sorted([d for d in os.listdir(tree_dir) if os.path.isdir(os.path.join(tree_dir, d))])
    
    for folder in subdirs:
        folder_path = os.path.join(tree_dir, folder)
        meta_file = os.path.join(folder_path, "meta.json")
        meta = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {meta_file}: {e}")
        
        stage_info = {
            "folder": folder,
            "path": folder_path,
            "title": meta.get("title", folder),
            "sub": meta.get("sub", "未定义阶段描述"),
            "badge": meta.get("badge", "常规阶段"),
            "badge_type": meta.get("badge_type", "default")
        }
        stages.append(stage_info)
    return stages


def render_poster(
    tree_dir: str = "assets/poster_tree",
    output_path: str = "docs/04_experiments/imgs/calligdit_evolution_master_poster.png",
    cell_size: int = 200,
    main_title: str = "马良 (Callig-DiT) 跨代演进与多阶段全景对比汇报海报",
    subtitle: str = "全景演进脉络：初代骨架(v10b) → 50k高保真清洗(v13) → 显式SkelNet形变(v21/v23) → 风格层级路由(v66) → 3x对称增广+C2OT旗舰(v68)",
    summary_tag: str = "核心演进结论：从早期字表强过拟合/粗糙笔触，经由SkelNet解耦形变与层级路由，最终在v68实现名家特异形变与真实碑帖墨韵的完美融合！"
):
    print(f"=== 正在从目录树 [{tree_dir}] 渲染全景海报 ===")
    columns = load_columns(tree_dir)
    stages = load_stages(tree_dir)

    n_cols = len(columns)
    n_rows = len(stages)
    print(f"检测到 {n_cols} 个样本列, {n_rows} 个阶段行:")
    for s in stages:
        print(f"  • [{s['folder']}] -> {s['title']} ({s['badge']})")

    # 布局参数定义
    row_header_w = 340
    top_header_h = 145
    col_header_h = 60
    padding = 16
    gap = 8

    canvas_w = row_header_w + n_cols * cell_size + (n_cols - 1) * gap + padding * 2
    canvas_h = top_header_h + col_header_h + n_rows * cell_size + (n_rows - 1) * gap + padding * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 249, 252))
    draw = ImageDraw.Draw(canvas)

    # 字体准备
    font_main_title = find_font(28, bold=True)
    font_sub_title = find_font(15)
    font_tag = find_font(15, bold=True)
    font_row_title = find_font(17, bold=True)
    font_row_sub = find_font(12)
    font_row_badge = find_font(12, bold=True)
    font_col_char = find_font(20, bold=True)
    font_col_desc = find_font(13)
    font_placeholder = find_font(13, bold=True)

    # 1. 顶部 Header 绘制
    draw.rectangle([0, 0, canvas_w, top_header_h], fill=(24, 28, 38))
    draw.text((padding + 12, 22), main_title, font=font_main_title, fill=(255, 255, 255))
    draw.text((padding + 14, 70), subtitle, font=font_sub_title, fill=(180, 195, 215))
    draw.text((padding + 14, 98), summary_tag, font=font_tag, fill=(255, 212, 100))

    # 2. 列标题卡片绘制
    y_col_hdr = top_header_h + padding
    for c_idx, col in enumerate(columns):
        x = row_header_w + padding + c_idx * (cell_size + gap)
        draw.rectangle(
            [x, y_col_hdr, x + cell_size, y_col_hdr + col_header_h - 4],
            fill=(234, 238, 245),
            outline=(210, 216, 228),
            width=1
        )
        tag_char = f"{col.get('char', '')}"
        tag_desc = f"{col.get('callig', '')} · {col.get('script', '')}书"
        draw.text((x + 14, y_col_hdr + 8), tag_char, font=font_col_char, fill=(20, 25, 35))
        draw.text((x + 48, y_col_hdr + 14), tag_desc, font=font_col_desc, fill=(80, 95, 115))
        draw.text((x + 14, y_col_hdr + 36), f"Sample #{col.get('idx', c_idx)}", font=find_font(11), fill=(130, 140, 155))

    # 3. 逐行绘制各阶段
    y_start = y_col_hdr + col_header_h
    for r_idx, stage in enumerate(stages):
        y = y_start + r_idx * (cell_size + gap)
        b_theme = BADGE_PALETTE.get(stage["badge_type"], BADGE_PALETTE["default"])

        # 3.1 左侧行头卡片
        draw.rectangle(
            [padding, y, row_header_w - 12, y + cell_size],
            fill=(255, 255, 255),
            outline=b_theme["border"],
            width=2
        )
        draw.text((padding + 16, y + 20), stage["title"], font=font_row_title, fill=(20, 30, 45))
        draw.text((padding + 16, y + 55), stage["sub"], font=font_row_sub, fill=(100, 110, 125))

        # Badge 徽章
        badge_y1 = y + 120
        badge_y2 = y + 165
        draw.rectangle(
            [padding + 14, badge_y1, row_header_w - 26, badge_y2],
            fill=b_theme["box_bg"],
            outline=b_theme["border"],
            width=1
        )
        draw.text((padding + 22, badge_y1 + 10), stage["badge"], font=font_row_badge, fill=b_theme["text"])

        # 3.2 绘制该行各个单元格图片
        for c_idx, col in enumerate(columns):
            x = row_header_w + padding + c_idx * (cell_size + gap)
            idx = col.get("idx", c_idx)
            
            # 支持多种命名规则查找: 00.png, 0.png, g0.png, gt0.png
            possible_names = [f"{idx:02d}.png", f"{idx}.png", f"g{idx}.png", f"gt{idx}.png"]
            img_path = None
            for nm in possible_names:
                p = os.path.join(stage["path"], nm)
                if os.path.exists(p):
                    img_path = p
                    break

            if img_path:
                im = Image.open(img_path).convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                canvas.paste(im, (x, y))
            else:
                # 缺失图像时的增量占位卡片
                im_ph = Image.new("RGB", (cell_size, cell_size), (243, 245, 248))
                dim = ImageDraw.Draw(im_ph)
                dim.rectangle([4, 4, cell_size - 5, cell_size - 5], fill=(236, 239, 244), outline=(220, 225, 232))
                dim.text((cell_size // 5, cell_size // 2 - 12), "【待增量更新】", font=font_placeholder, fill=(150, 160, 175))
                canvas.paste(im_ph, (x, y))

            # 格子边框强调
            if stage["badge_type"] == "gt":
                border_col, border_w = (200, 40, 40), 2
            elif stage["badge_type"] == "accent":
                border_col, border_w = (40, 130, 220), 2
            else:
                border_col, border_w = (220, 225, 232), 1
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=border_col, width=border_w)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path, quality=95, optimize=True)
    sz_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"🎉 汇报全景 Poster 已成功渲染落盘至: {output_path}")
    print(f"   尺寸: {canvas_w}x{canvas_h} px | 文件大小: {sz_mb:.2f} MB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Callig-DiT Evolution Poster from Directory Tree")
    parser.add_argument("--tree-dir", type=str, default="assets/poster_tree", help="阶段目录树路径")
    parser.add_argument("--output", type=str, default="docs/04_experiments/imgs/calligdit_evolution_master_poster.png", help="输出图片路径")
    parser.add_argument("--cell-size", type=int, default=200, help="单元格图像边长尺寸")
    parser.add_argument("--title", type=str, default="马良 (Callig-DiT) 跨代演进与多阶段全景对比汇报海报", help="海报大标题")
    args = parser.parse_args()

    render_poster(
        tree_dir=args.tree_dir,
        output_path=args.output,
        cell_size=args.cell_size,
        main_title=args.title
    )
