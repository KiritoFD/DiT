#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""large_eval 三方对比图: 每行 = diffonly 生成 | struct 生成 | GT。"""
import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                    "s6_report", "large_eval")
N = 16
CELL = 200
PAD = 4
HEAD = 34

def cell(path):
    im = Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    return im

rows = []
for i in range(N):
    d = cell(os.path.join(BASE, "diffonly_s195000", "latest", f"sample{i}.png"))
    s = cell(os.path.join(BASE, "struct_s125000", "latest", f"sample{i}.png"))
    g = cell(os.path.join(BASE, "diffonly_s195000", "latest", f"gt{i}.png"))
    rows.append((d, s, g))

W = PAD * 4 + CELL * 3
H = HEAD + (CELL + PAD) * N + PAD
canvas = Image.new("RGB", (W, H), (18, 20, 26))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
except Exception:
    font = ImageFont.load_default()
labels = [("diffonly@195k", (120, 220, 255)), ("struct@125k", (255, 170, 120)),
          ("GT", (180, 255, 180))]
for c, (txt, col) in enumerate(labels):
    x = PAD + c * (CELL + PAD)
    draw.text((x + 6, 6), txt, font=font, fill=col)

for r, (d, s, g) in enumerate(rows):
    y = HEAD + r * (CELL + PAD)
    for c, im in enumerate((d, s, g)):
        canvas.paste(im, (PAD + c * (CELL + PAD), y))

out = os.path.join(BASE, "compare_diffonly_vs_struct.png")
canvas.save(out)
print("saved", out, canvas.size)
