import os
from PIL import Image, ImageDraw, ImageFont

items = [
    {"name": "柳公权 楷书「連」", "idx": 46},
    {"name": "王羲之 楷书「旨」", "idx": 80},
    {"name": "王羲之 行书「好」", "idx": 88},
    {"name": "颜真卿 楷书「其」", "idx": 172},
]

n_cols = len(items)
cell_w, cell_h = 256, 256
pad = 8
left_header_w = 400
top_header_h = 70

canvas_w = left_header_w + n_cols * cell_w + (n_cols + 1) * pad
canvas_h = top_header_h + 4 * cell_h + 5 * pad

canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 19, 22))
draw = ImageDraw.Draw(canvas)

# Fonts
try:
    font_col = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    font_row = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    font_desc = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
except Exception:
    font_col = font_row = font_desc = ImageFont.load_default()

# Top Header: Column Titles
for c_i, it in enumerate(items):
    x = left_header_w + pad + c_i * (cell_w + pad)
    text = it["name"]
    # Centered text in cell width
    bbox = draw.textbbox((0, 0), text, font=font_col)
    tw = bbox[2] - bbox[0]
    tx = x + (cell_w - tw) // 2
    draw.text((tx, 22), text, fill=(240, 243, 250), font=font_col)

# Row Configurations
row_defs = [
    {
        "title": "Row 1: 【输入条件】标准宋体",
        "sub": "宋体印刷字骨架 (Standard Font)\n作为空间输入几何先验",
        "color": (90, 160, 255),
        "prefix": "std",
        "bg": (28, 35, 48)
    },
    {
        "title": "Row 2: 【参考方案】墨意/墨韵 Moyi",
        "sub": "Moyun 12ch RF (复现 20,000步)\n无空间骨架注入, 仅类条件ID\n(从高斯白噪声全盲记忆生成)",
        "color": (255, 95, 95),
        "prefix": "moyi",
        "bg": (48, 28, 28)
    },
    {
        "title": "Row 3: 【我们方案】Callig-DiT",
        "sub": "单阶段纯标准字条件 (同期 22,500步)\n空间几何特征注入 (Cross-Attn/AdaLN)\n(标准印刷体->名家墨迹风格形变)",
        "color": (80, 230, 140),
        "prefix": "our",
        "bg": (25, 45, 32)
    },
    {
        "title": "Row 4: 【真实真迹】古代碑帖",
        "sub": "古代名家书法碑帖真迹 (Ground-Truth)\n(真实历史拓片/墨迹基准)",
        "color": (255, 215, 80),
        "prefix": "gt",
        "bg": (45, 42, 25)
    },
]

# Draw Rows
for r_i, rdef in enumerate(row_defs):
    y = top_header_h + pad + r_i * (cell_h + pad)
    
    # Left Header Box
    draw.rectangle(
        [pad, y, left_header_w - pad, y + cell_h],
        fill=rdef["bg"],
        outline=rdef["color"],
        width=2
    )
    draw.text((pad + 18, y + 28), rdef["title"], fill=rdef["color"], font=font_row)
    draw.text((pad + 18, y + 78), rdef["sub"], fill=(210, 215, 225), font=font_desc)
    
    # Image Cells
    for c_i, it in enumerate(items):
        idx = it["idx"]
        x = left_header_w + pad + c_i * (cell_w + pad)
        
        img_file = f"exp/h2h_cache/{rdef['prefix']}_{idx}.png"
        im = Image.open(img_file).convert("RGB")
        im = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        canvas.paste(im, (x, y))
        
        # Draw clean border around each image cell
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(55, 60, 70), width=1)

out_poster = "exp/moyi_vs_calligdit_head_to_head.png"
canvas.save(out_poster, quality=95)
print(f"Saved Head-to-Head Panoramic Comparison: {out_poster} ({canvas_w}x{canvas_h})")
