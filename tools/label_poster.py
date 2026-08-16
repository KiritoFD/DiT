#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成带字标签的 poster, 每列标出 show5 对应字, 便于核对 GT 行对齐。"""
import os, csv, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
show5 = os.path.join(HERE, "show5_eval.csv")
poster = os.path.join(HERE, "eval_poster.png")

CELL = 224
# 读 5 个字
rows = list(csv.DictReader(open(show5, encoding="utf-8")))[:5]
chars = [r["character"] for r in rows]
ids = [os.path.basename(r["image_path"])[:-4] for r in rows]

img = Image.open(poster).convert("RGB")
W, H = img.size
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
except Exception:
    font = ImageFont.load_default()
# 顶部加一行标注: 每列的字 + id
for i in range(5):
    x0 = CELL * 1 + i * CELL * 3  # 第一列是步号? 实际 post poster 里每 step 占 3*CELL
    # 直接在顶部中间画标签(虚线一行)
    d.text((x0, 4), f"{chars[i]}({ids[i]})", fill=(255, 220, 80), font=font)
# 保存
out = os.path.join(HERE, "eval_poster_labeled.png")
img.save(out)
print(f"labeled poster -> {out}")
print(f"GT 行(最后一行) 每列字: {chars}")
print(f"对应 image ids: {ids}")
