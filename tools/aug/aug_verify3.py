# -*- coding: utf-8 -*-
"""aug_verify3.py — 量化 v3 变体差异 + 强案例放大条"""
import csv
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/5script/train_fame3_clean_v8.csv"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, f"{ROOT}/tools/aug")
from aug_renders_v3 import make_variant  # noqa: E402

rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
rng = random.Random(7)
picks = rng.sample(range(len(rows)), 50)

print(f"{'字':>2} {'kind':>2} {'meanAbsDiff':>11} {'px>16':>7}")
stats = {k: [] for k in "abc"}
strong = {}
for idx in picks:
    r = rows[idx]
    im = Image.open(os.path.join(ROOT, r["image_path"])).convert("L")
    o = np.asarray(im, dtype=np.int16)
    rng2 = random.Random(10_000 + idx)
    for kind in "abc":
        v, info = make_variant(im, rng2, kind)
        d = np.abs(np.asarray(v, dtype=np.int16) - o)
        m, p = d.mean(), (d > 16).mean() * 100
        stats[kind].append((m, p))
        if kind == "b" and info.get("delta"):
            strong.setdefault(r["character"], (idx, info))
for kind in "abc":
    ms = np.array(stats[kind])
    print(f"{kind}: meanAbsDiff={ms[:, 0].mean():6.2f}  px>16={ms[:, 1].mean():5.1f}%  "
          f"(max {ms[:, 0].max():.1f})")

# 放大条: 每例 [原图 | b变体] 中心 128crop -> 192px, 4 例
FT = "/tmp/simhei.ttf"
f2 = ImageFont.truetype(FT, 12)
exs = list(strong.items())[:4]
cv = Image.new("RGB", (len(exs) * (192 * 2 + 14) + 10, 218), "white")
dr = ImageDraw.Draw(cv)
for j, (ch, (idx, info)) in enumerate(exs):
    r = rows[idx]
    im = Image.open(os.path.join(ROOT, r["image_path"])).convert("L")
    rng2 = random.Random(10_000 + idx)
    v, _ = make_variant(im, rng2, "b")
    x0 = 5 + j * (192 * 2 + 14)
    for k, img in enumerate((im, v)):
        crop = img.crop((64, 64, 192, 192)).resize((192, 192), Image.NEAREST)
        cv.paste(crop, (x0 + k * 194, 22))
    dr.text((x0, 4), f"{ch} {r['calligrapher']} w={info['w']}→{info['delta']:+.1f}px", font=f2, fill="black")
cv.save("/tmp/aug_zoom_b.png")
print("saved /tmp/aug_zoom_b.png")
