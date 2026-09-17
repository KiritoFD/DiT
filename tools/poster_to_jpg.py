# -*- coding: utf-8 -*-
"""poster_to_jpg.py — poster 转小尺寸 JPEG 便于回传查看。"""
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W = int(sys.argv[1]) if len(sys.argv) > 1 else 900
JOBS = sys.argv[2:]
for i in range(0, len(JOBS), 2):
    src, dst = JOBS[i], JOBS[i + 1]
    if not os.path.exists(src):
        print(f"MISS {src}")
        continue
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im = im.resize((W, max(1, int(h * W / w))), Image.LANCZOS)
    im.save(dst, quality=80)
    print(f"{src} {w}x{h} -> {im.size} {os.path.getsize(dst)//1024}KB {dst}")
