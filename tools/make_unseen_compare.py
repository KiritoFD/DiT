#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""unseen 三方对比图: 每行 = diffonly 生成 | struct 生成 | GT。"""
import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.join("docs", "s6_report", "large_eval")
N, CELL, PAD, HEAD = 16, 200, 4, 34


def cell(p):
    return Image.open(p).convert("RGB").resize((CELL, CELL), Image.LANCZOS)


rows = [(cell(os.path.join(BASE, "diffonly_s195000_unseen", "latest", f"sample{i}.png")),
         cell(os.path.join(BASE, "struct_s125000_unseen", "latest", f"sample{i}.png")),
         cell(os.path.join(BASE, "diffonly_s195000_unseen", "latest", f"gt{i}.png")))
        for i in range(N)]

W, H = PAD * 4 + CELL * 3, HEAD + (CELL + PAD) * N + PAD
cv = Image.new("RGB", (W, H), (18, 20, 26))
d = ImageDraw.Draw(cv)
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
for c, (t, col) in enumerate([("diffonly@195k unseen", (120, 220, 255)),
                              ("struct@125k unseen", (255, 170, 120)),
                              ("GT", (180, 255, 180))]):
    d.text((PAD + c * (CELL + PAD) + 6, 6), t, font=font, fill=col)
for r, imgs in enumerate(rows):
    y = HEAD + r * (CELL + PAD)
    for c, im in enumerate(imgs):
        cv.paste(im, (PAD + c * (CELL + PAD), y))
out = os.path.join(BASE, "compare_unseen.png")
cv.save(out)
print("saved", out, cv.size)
