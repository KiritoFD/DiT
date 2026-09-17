import sys
from PIL import Image
import os

# usage: python resize.py <src> <dst> <max_w> <max_h>
src, dst, max_w, max_h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
im = Image.open(src)
w, h = im.size
print("orig", w, h)
s = min(max_w / w, max_h / h, 1.0)
nw, nh = max(1, int(w * s)), max(1, int(h * s))
im2 = im.convert("RGB").resize((nw, nh), Image.LANCZOS)
im2.save(dst, quality=88)
print("saved", dst, nw, nh, os.path.getsize(dst))
