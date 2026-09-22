"""生成「跨 ckpt 都差」样本的审阅 poster（GT | std g | 生成图）。

## 目的
目检这三者的关系：
- GT vs std g 差异大 -> 可能是标注错配（g 是另一个字）
- GT vs std g 差异小 -> 是"模型写不好"，不是数据问题
"""
import csv
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")

N = int(os.environ.get("N", "30"))
SZ = 160
rows = list(csv.DictReader(open("assets/eval_low_ssim.csv",
                                encoding="utf-8")))[:N]

# 找一张生成图（用最好的 ckpt v13_base_50k 的 eval 存图）
gen_dirs = glob.glob("assets/results/v13_base_50k/*/eval_samples_ctrl/*/strict50")
gen_map = {}
if gen_dirs:
    gd = sorted(gen_dirs)[-1]
    for f in glob.glob(os.path.join(gd, "g*.png")):
        idx = int(os.path.basename(f)[1:-4])
        gen_map[idx] = f
print(f"  生成图目录: {gen_dirs[-1] if gen_dirs else '无'}")
print(f"  找到 {len(gen_map)} 张生成图")

try:
    font = ImageFont.truetype("_fonts/msyh.ttc", 18)
except Exception:
    font = ImageFont.load_default()

COLS = 6
ROWS = (N + COLS - 1) // COLS
CELL_W, CELL_H = SZ, SZ + 26
canvas = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H * 3 + 30), "white")
d = ImageDraw.Draw(canvas)
d.text((6, 4), "行1=GT(书法)  行2=std g(印刷骨架)  行3=模型生成(v13_base)",
       font=font, fill="black")

for i, r in enumerate(rows):
    col, row = i % COLS, i // COLS
    x0, y0 = col * CELL_W, 30 + row * CELL_H * 3
    iid = int(r["img_id"])
    # GT
    gt_p = f"data/50k/imgs/{iid:06d}.png"
    std_p = f"data/50k/std/{iid:06d}.png"
    idx = int(r["idx"])
    for k, p in enumerate((gt_p, std_p, gen_map.get(idx))):
        yy = y0 + k * CELL_H
        if p and os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((SZ, SZ), Image.BICUBIC)
            canvas.paste(im, (x0, yy))
        else:
            d.rectangle([x0, yy, x0 + SZ, yy + SZ], outline="lightgray")
        if k == 2:
            d.text((x0 + 3, yy + SZ + 2),
                   f"{r['char']} {r['script']} {r['mean_ssim']}",
                   font=font, fill="black")

out = "assets/poster_low_ssim.png"
canvas.save(out)
print(f"  ✓ -> {out}  尺寸={canvas.size}")
