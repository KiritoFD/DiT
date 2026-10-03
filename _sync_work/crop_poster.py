#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裁 poster 左侧一段 + 前 3 行 (input/step/GT), 放大 2x, 便于肉眼判。"""
import sys

from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
frac_w = float(sys.argv[3]) if len(sys.argv) > 3 else 0.14
frac_h = float(sys.argv[4]) if len(sys.argv) > 4 else 0.55
im = Image.open(src)
W, H = im.size
box = (0, 0, int(W * frac_w), int(H * frac_h))
c = im.crop(box)
c = c.resize((c.size[0] * 2, c.size[1] * 2), Image.LANCZOS)
c.save(dst)
print(f"{src} {W}x{H} -> {dst} {c.size[0]}x{c.size[1]}")
