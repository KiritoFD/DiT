# -*- coding: utf-8 -*-
"""shrink_poster.py — 把 poster 缩小以便下载查看。"""
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JOBS = sys.argv[1:]
for i in range(0, len(JOBS), 2):
    src, dst = JOBS[i], JOBS[i + 1]
    if not os.path.exists(src):
        print(f"MISS {src}")
        continue
    im = Image.open(src)
    w, h = im.size
    f = max(1, int(round(max(w, h) / 1400.0)))
    if f > 1:
        im = im.resize((w // f, h // f), Image.LANCZOS)
    im.save(dst)
    print(f"{src}  {w}x{h} -> {im.size[0]}x{im.size[1]}  {dst}")
