# -*- coding: utf-8 -*-
"""aug_preview2.py — v2 预览: 随机 50 张, [原图 | _a 墨密度 | _b 去噪+选择性加粗 | _c shift/rotate]"""
import csv
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/assets/train_fame3_clean_v8.csv"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, f"{ROOT}/tools/aug")
from aug_renders_v2 import make_variant  # noqa: E402

rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
rng = random.Random(7)
picks = rng.sample(range(len(rows)), 50)

FT = "/tmp/simhei.ttf"
CELL = 96
PER_ROW = 4  # 每行 4 个样本 × 4 列
NROW = (50 + PER_ROW - 1) // PER_ROW
cv = Image.new("RGB", (PER_ROW * (CELL * 4 + 8) + 10, NROW * (CELL + 26) + 40), "white")
dr = ImageDraw.Draw(cv)
f2 = ImageFont.truetype(FT, 11)
dr.text((8, 4), "aug v2 preview: 每组 [原图 | _a 墨密度×0.8-1.25 | _b 去噪+细笔画补粗 | _c 平移±4px+旋转±1.5°]  ×50",
        font=ImageFont.truetype(FT, 14), fill="black")
for j, idx in enumerate(picks):
    r = rows[idx]
    im = Image.open(os.path.join(ROOT, r["image_path"])).convert("L")
    rng2 = random.Random(10_000 + idx)
    infos = {}
    outs = [im]
    for kind in ("a", "b", "c"):
        v, info = make_variant(im, rng2, kind)
        outs.append(v)
        infos[kind] = info
    rr, c = divmod(j, PER_ROW)
    x0 = 5 + c * (CELL * 4 + 8)
    y0 = 30 + rr * (CELL + 26)
    for k, v in enumerate(outs):
        cv.paste(v.resize((CELL, CELL)), (x0 + k * (CELL + 2), y0))
    b, cc = infos["b"], infos["c"]
    lab = f"{r['character']} {r['calligrapher']}"
    if b:
        lab += f" | w={b['width']} {'+' + str(b['thicken_passes']) + 'px' if b['thicken_passes'] else '不补'}"
    lab += f" | rot{cc['angle']:+.1f} sh({cc['shift'][0]:+d},{cc['shift'][1]:+d})"
    dr.text((x0, y0 + CELL + 2), lab, font=f2, fill="black")
cv.save("/tmp/aug_preview2.png")
print("saved /tmp/aug_preview2.png")
