# -*- coding: utf-8 -*-
"""aug_preview3.py — 预览: 50 张, [原图 | _b1 | _b2 | _b3]"""
import csv
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/5script/train_fame3_clean_v8.csv"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, f"{ROOT}/tools/aug")
from aug_renders_v3 import make_variant  # noqa: E402

rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
rng = random.Random(7)
picks = rng.sample(range(len(rows)), 50)

FT = "/tmp/simhei.ttf"
CELL = 96
PER_ROW = 4
NROW = (50 + PER_ROW - 1) // PER_ROW
cv = Image.new("RGB", (PER_ROW * (CELL * 4 + 8) + 10, NROW * (CELL + 26) + 40), "white")
dr = ImageDraw.Draw(cv)
f2 = ImageFont.truetype(FT, 11)
dr.text((8, 4), "aug v3.3: [原图 | _b1 K0.6/cap35% | _b2 K1.2/cap50% | _b3 K2.0/cap65%]  二值加粗/收窄 ×50",
        font=ImageFont.truetype(FT, 14), fill="black")
for j, idx in enumerate(picks):
    r = rows[idx]
    im = Image.open(os.path.join(ROOT, r["image_path"])).convert("L")
    rng2 = random.Random(10_000 + idx)
    outs, infos = [im], {}
    for kind in ("b1", "b2", "b3"):
        v, info = make_variant(im, rng2, kind)
        outs.append(v)
        infos[kind] = info
    rr, c = divmod(j, PER_ROW)
    x0 = 5 + c * (CELL * 4 + 8)
    y0 = 30 + rr * (CELL + 26)
    for k, v in enumerate(outs):
        cv.paste(v.resize((CELL, CELL)), (x0 + k * (CELL + 2), y0))
    b1, b2, b3 = infos["b1"], infos["b2"], infos["b3"]
    lab = f"{r['character']} {r['calligrapher']} w={b1['w']:.1f}"
    lab += f" | {b1['delta']:+d}/{b2['delta']:+d}/{b3['delta']:+d}px"
    dr.text((x0, y0 + CELL + 2), lab, font=f2, fill="black")
cv.save("/tmp/aug_preview3.png")
print("saved /tmp/aug_preview3.png")
