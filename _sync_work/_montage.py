# -*- coding: utf-8 -*-
"""_montage.py — 按数据源/分类拼缩略图网格, 用于肉眼判断底色.

用法: python _montage.py <csv> <filter> <n> <out.png>
  filter: all | dark  (dark = 读 /tmp/polarity_v2_base.csv 中非 ok 的图)
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV = sys.argv[1]
FILTER = sys.argv[2] if len(sys.argv) > 2 else "all"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
OUT = sys.argv[4] if len(sys.argv) > 4 else "/root/Workspace/xy/DiT/_otout_montage.png"
SRC = sys.argv[5] if len(sys.argv) > 5 else ""      # 数据源子串过滤
CELL = 96
COLS = 8

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
if SRC:
    rows = [r for r in rows if SRC in r.get("image_path", "")]
if FILTER == "dark" and os.path.exists("/tmp/polarity_v2_base.csv"):
    dk = [r["image_path"] for r in csv.DictReader(open("/tmp/polarity_v2_base.csv",
                                                        encoding="utf-8"))]
    rows = [r for r in rows if r["image_path"] in set(dk)]
print(f"[montage] {len(rows)} candidates (filter={FILTER}, src={SRC})", flush=True)

sel = rows[:N]
ncol = COLS
nrow = (len(sel) + ncol - 1) // ncol
canvas = Image.new("RGB", (ncol * CELL, nrow * CELL), (128, 0, 0))
d = ImageDraw.Draw(canvas)
for i, r in enumerate(sel):
    p = r["image_path"]
    try:
        im = Image.open(p).convert("L").resize((CELL, CELL))
        a = np.asarray(im)
        canvas.paste(Image.fromarray(a).convert("RGB"), ((i % ncol) * CELL, (i // ncol) * CELL))
    except Exception as e:
        print("  fail", p, e)
# 底部不写字(避免字体问题), 直接保存
canvas.save(OUT)
print(f"-> {OUT}  ({nrow}x{ncol} cells, {len(sel)} imgs)", flush=True)

# 同时打印每张的统计, 便于和视觉对照
for i, r in enumerate(sel[:16]):
    p = r["image_path"]
    a = np.asarray(Image.open(p).convert("L"))
    print(f"  #{i:02d} med={np.median(a):5.1f} mean={a.mean():5.1f} "
          f"ink={float((a<128).mean()):.3f} bright={float((a>128).mean()):.3f} "
          f"{os.path.basename(p)} char={r.get('character','')} {r.get('calligrapher','')}")
