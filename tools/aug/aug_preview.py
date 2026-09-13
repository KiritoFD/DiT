# -*- coding: utf-8 -*-
"""aug_preview.py — 渲染扰动预览: 随机 50 张, [原图 | _a 墨密度 | _b 笔画宽度] 拼图"""
import csv
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/assets/train_fame3_clean_v8.csv"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, f"{ROOT}/tools/aug")
from aug_renders_v1 import make_variant  # noqa: E402

rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
rng = random.Random(7)
picks = rng.sample(range(len(rows)), 50)

FT = "/tmp/simhei.ttf"
CELL = 96
PER_ROW = 5  # 每行 5 个样本 × 3 列
cv = Image.new("RGB", (PER_ROW * (CELL * 3 + 8) + 10, ((50 + PER_ROW - 1) // PER_ROW) * (CELL + 26) + 40), "white")
dr = ImageDraw.Draw(cv)
f2 = ImageFont.truetype(FT, 11)
dr.text((8, 4), "aug preview: 每组 [原图 | _a 墨密度(gamma/contrast/blur) | _b 笔画宽度(±1px)]  ×50", font=ImageFont.truetype(FT, 14), fill="black")
for j, idx in enumerate(picks):
    r = rows[idx]
    im = Image.open(os.path.join(ROOT, r["image_path"])).convert("L")
    rng2 = random.Random(10_000 + idx)
    a = make_variant(im, rng2, "a")
    b = make_variant(im, rng2, "b")
    rr, c = divmod(j, PER_ROW)
    x0 = 5 + c * (CELL * 3 + 8)
    y0 = 30 + rr * (CELL + 26)
    for k, v in enumerate((im, a, b)):
        cv.paste(v.resize((CELL, CELL)), (x0 + k * (CELL + 2), y0))
    dr.text((x0, y0 + CELL + 2), f"{r['character']} {r['calligrapher']}", font=f2, fill="black")
cv.save("/tmp/aug_preview.png")
print("saved /tmp/aug_preview.png")
