# -*- coding: utf-8 -*-
"""render_visual_poster.py —— 制作 24 字横向对照海报 (STD | GT | v84A | v85A | v85C | v85E@15k | v85E@30k)
并量化分析笔画宽度、墨量与高频边缘变化（逐项回答专家诊断 2 的四大疑问）。
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from src.eval.style.bank import resolve
from src.eval.style.brush import features_from_image

ROOTS = [".", "data", "exp-std"]
CSV_PATH = "exp-std/csv/eval200_fixed.csv"

# 各模型目录
MODELS = [
    ("STD", None),
    ("GT真迹", None),
    ("v84A@200k", "assets/style_eval_src/step0200000"),
    ("v85A@30k", "assets/style_eval_src/v85A_step0030000"),
    ("v85C@30k", "assets/style_eval_src/v85C_step0030000"),
    ("v85E@15k", "assets/style_eval_src/v85E_step0015000"),
    ("v85E@30k", "assets/style_eval_src/v85E_step0030000"),
]

def load_rows():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    return rows

def select_diverse_chars(rows, n=24):
    """挑选覆盖不同书家、书体、笔画复杂度的 24 个代表字"""
    by_writer = defaultdict(list)
    for idx, r in enumerate(rows):
        by_writer[r["calligrapher"]].append((idx, r))
    
    selected = []
    # 轮流从每个书家抽取不同书体/复杂度的字
    writers = sorted(by_writer.keys())
    while len(selected) < n:
        for w in writers:
            if by_writer[w] and len(selected) < n:
                selected.append(by_writer[w].pop(0))
    return selected

def main():
    rows = load_rows()
    selected = select_diverse_chars(rows, n=24)
    print(f"已选取 {len(selected)} 个代表字符进行逐字深度对照分析")

    # 统计数据
    stats = defaultdict(lambda: {"width_p50": [], "ink_frac": [], "edge_mag": []})

    img_size = 256
    pad = 10
    header_h = 60
    label_w = 140
    cols = len(MODELS)
    
    total_w = label_w + cols * (img_size + pad) + pad
    total_h = header_h + len(selected) * (img_size + pad) + pad
    
    poster = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(poster)
    
    # 字体尝试
    font_path = "C:/Windows/Fonts/simhei.ttf" if os.path.exists("C:/Windows/Fonts/simhei.ttf") else "tools/fonts/SimHei.ttf"
    try:
        font_large = ImageFont.truetype(font_path, 22)
        font_small = ImageFont.truetype(font_path, 16)
    except Exception:
        font_large = font_small = None

    # 画列头
    for col_idx, (m_name, _) in enumerate(MODELS):
        x = label_w + col_idx * (img_size + pad) + img_size // 2
        draw.text((x, 20), m_name, fill=(30, 30, 30), font=font_large, anchor="mt")

    # 逐行画字并统计
    for row_idx, (orig_idx, r) in enumerate(selected):
        y = header_h + row_idx * (img_size + pad)
        ch = r["character"]
        writer = r["calligrapher"]
        script = r["script"]
        
        # 行标
        label_text = f"{ch}\n{writer}\n({script})"
        draw.text((pad + 10, y + img_size // 2), label_text, fill=(40, 40, 40), font=font_small, anchor="lm")
        
        for col_idx, (m_name, m_dir) in enumerate(MODELS):
            x = label_w + col_idx * (img_size + pad)
            
            # 读取图片
            if m_name == "STD":
                p = resolve(r.get("std_path", "").replace("\\", "/"), ROOTS)
            elif m_name == "GT真迹":
                p = resolve(r.get("image_path", "").replace("\\", "/"), ROOTS)
            else:
                p = os.path.join(m_dir, f"g{orig_idx}.png")
            
            if p and os.path.isfile(p):
                im = Image.open(p).convert("RGB")
                if im.size != (img_size, img_size):
                    im = im.resize((img_size, img_size), Image.BICUBIC)
                poster.paste(im, (x, y))
                draw.rectangle([x, y, x + img_size, y + img_size], outline=(200, 200, 200), width=1)
                
                # 统计特征
                feat = features_from_image(p)
                if feat.ok:
                    w_stat = feat.width_stats()
                    stats[m_name]["width_p50"].append(w_stat["p50"])
                    stats[m_name]["ink_frac"].append(feat.scalars.get("ink_frac", 0.0))
                    stats[m_name]["edge_mag"].append(float(feat.edge.mean()) if feat.edge.size else 0.0)

    out_img = "ANALYSIS/v85_poster_comparison.png"
    os.makedirs("ANALYSIS", exist_ok=True)
    poster.save(out_img, quality=95)
    print(f"✓ 海报已生成并保存至: {out_img}")

    print("\n" + "="*80)
    print("【24 字代表样本 笔墨物理微观统计分析】")
    print(f"{'模型名称':<15} | {'笔画中位宽度 (px)':<18} | {'墨量覆盖占比 (ink_frac)':<22} | {'边缘梯度强度 (edge_mag)':<18}")
    print("-" * 80)
    for m_name, _ in MODELS:
        s = stats[m_name]
        w_m = np.mean(s["width_p50"]) if s["width_p50"] else float("nan")
        ink_m = np.mean(s["ink_frac"]) if s["ink_frac"] else float("nan")
        edge_m = np.mean(s["edge_mag"]) if s["edge_mag"] else float("nan")
        print(f"{m_name:<15} | {w_m:<18.2f} | {ink_m*100:<20.2f}% | {edge_m:<18.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
