import os, sys
import subprocess
from PIL import Image, ImageDraw, ImageFont
import numpy as np

os.makedirs("exp/h2h_cache", exist_ok=True)

# 8 Matched Samples
items = [
    {
        "name": "柳公权 楷書「連」",
        "our_idx": 46,
        "img_id": "010042.png",
        "moyi_file": "moyi_g6_柳公权_楷_連.png",
    },
    {
        "name": "王羲之 楷書「旨」",
        "our_idx": 80,
        "img_id": "017513.png",
        "moyi_file": "moyi_g10_王羲之_楷_旨.png",
    },
    {
        "name": "王羲之 行書「好」",
        "our_idx": 88,
        "img_id": "017687.png",
        "moyi_file": "moyi_g11_王羲之_行_好.png",
    },
    {
        "name": "米芾 行書「墟」",
        "our_idx": 103,
        "img_id": "018038.png",
        "moyi_file": "moyi_g13_米芾_行_墟.png",
    },
    {
        "name": "赵孟頫 行書「匠」",
        "our_idx": 155,
        "img_id": "014290.png",
        "moyi_file": "moyi_g19_赵孟頫_行_匠.png",
    },
    {
        "name": "赵孟頫 隶書「盤」",
        "our_idx": 163,
        "img_id": "037234.png",
        "moyi_file": "moyi_g20_赵孟頫_隶_盤.png",
    },
    {
        "name": "颜真卿 楷書「其」",
        "our_idx": 172,
        "img_id": "020672.png",
        "moyi_file": "moyi_g21_颜真卿_楷_其.png",
    },
    {
        "name": "颜真卿 行書「憫」",
        "our_idx": 179,
        "img_id": "004219.png",
        "moyi_file": "moyi_g22_颜真卿_行_憫.png",
    },
]

print("1. Fetching images...")

# 1. Fetch from 4090: std, ours, gt
for it in items:
    idx = it["our_idx"]
    iid = it["img_id"]
    
    std_loc = f"exp/h2h_cache/std_{idx}.png"
    our_loc = f"exp/h2h_cache/our_{idx}.png"
    gt_loc  = f"exp/h2h_cache/gt_{idx}.png"
    
    if not os.path.exists(std_loc):
        subprocess.run(["scp", f"4090:/root/Workspace/xy/DiT/exp-std/runs_purestd/eval_samples_ctrl/eval200fix_input_g/g{idx}.png", std_loc], check=True)
    if not os.path.exists(our_loc):
        subprocess.run(["scp", f"4090:/root/Workspace/xy/DiT/exp-std/runs_purestd/eval_samples_ctrl/step0022500/eval200fix/g{idx}.png", our_loc], check=True)
    if not os.path.exists(gt_loc):
        subprocess.run(["scp", f"4090:/root/Workspace/xy/DiT/data/top10_style23/imgs/{iid}", gt_loc], check=True)

# 2. Fetch from 48: moyi
for it in items:
    idx = it["our_idx"]
    mfile = it["moyi_file"]
    moyi_loc = f"exp/h2h_cache/moyi_{idx}.png"
    if not os.path.exists(moyi_loc):
        subprocess.run(["scp", f"48:/home/ds/Workspace/moyi/results/moyi_top10_rf/eval_samples_step20000/{mfile}", moyi_loc], check=True)

print("2. Stitching Head-to-Head Panoramic Poster...")

# Dimensions
n_cols = len(items)
cell_w, cell_h = 256, 256
pad = 6
left_header_w = 340
top_header_h = 60

canvas_w = left_header_w + n_cols * cell_w + (n_cols + 1) * pad
canvas_h = top_header_h + 4 * cell_h + 5 * pad

canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 25, 28))
draw = ImageDraw.Draw(canvas)

# Fonts
try:
    font_col = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    font_row = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    font_desc = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
except Exception:
    font_col = font_row = font_desc = ImageFont.load_default()

# Draw Column Titles (Top Header)
for c_i, it in enumerate(items):
    x = left_header_w + pad + c_i * (cell_w + pad)
    text = it["name"]
    draw.text((x + 25, 18), text, fill=(230, 235, 245), font=font_col)

# Row Configurations
row_defs = [
    {
        "title": "Row 1: 【输入条件】",
        "sub": "标准宋体印刷字骨架\n(Standard Font Input)",
        "color": (120, 180, 255),
        "prefix": "std"
    },
    {
        "title": "Row 2: 【参考方案】墨意/墨韵",
        "sub": "Moyun 12ch RF (20,000步)\n无空间骨架注入, 仅类条件ID\n(从高斯纯噪声全盲生成)",
        "color": (255, 120, 120),
        "prefix": "moyi"
    },
    {
        "title": "Row 3: 【我们方案】Callig-DiT",
        "sub": "单阶段纯标准字条件 (22,500步)\n空间几何特征注入 (Cross-Attn)\n(标准印刷体->名家墨迹生成)",
        "color": (120, 255, 160),
        "prefix": "our"
    },
    {
        "title": "Row 4: 【真实真迹】",
        "sub": "古代名家书法碑帖真迹\n(Ground-Truth Real Calligraphy)\n(真实历史拓片/真迹基准)",
        "color": (255, 220, 100),
        "prefix": "gt"
    },
]

# Draw Left Header & Images
for r_i, rdef in enumerate(row_defs):
    y = top_header_h + pad + r_i * (cell_h + pad)
    
    # Left Header Box
    draw.rectangle([pad, y, left_header_w - pad, y + cell_h], fill=(32, 34, 38), outline=rdef["color"], width=2)
    draw.text((pad + 16, y + 25), rdef["title"], fill=rdef["color"], font=font_row)
    draw.text((pad + 16, y + 70), rdef["sub"], fill=(190, 195, 205), font=font_desc)
    
    # Cells
    for c_i, it in enumerate(items):
        idx = it["our_idx"]
        x = left_header_w + pad + c_i * (cell_w + pad)
        
        img_file = f"exp/h2h_cache/{rdef['prefix']}_{idx}.png"
        if os.path.exists(img_file):
            im = Image.open(img_file).convert("RGB")
            im = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(im, (x, y))
        else:
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(45, 45, 45))
            draw.text((x + 60, y + 110), "NOT FOUND", fill=(255, 80, 80), font=font_col)

# Save Final Poster
out_poster = "exp/moyi_vs_calligdit_head_to_head.png"
canvas.save(out_poster, quality=95)
print(f"✓ Head-to-Head Panoramic Poster successfully saved: {out_poster}")
