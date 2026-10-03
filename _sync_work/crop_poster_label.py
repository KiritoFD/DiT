#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裁出 poster 左侧标签带 + 前几列，确认到底包含哪些 step。"""
import sys
from PIL import Image

src = sys.argv[1]
out = sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 700
im = Image.open(src)
box = (0, 0, min(w, im.size[0]), im.size[1])
im.crop(box).save(out)
print(f'{src} 原尺寸 {im.size} -> 裁 {box[2]}x{box[3]} 到 {out}')
