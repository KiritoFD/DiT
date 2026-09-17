import sys
from PIL import Image

# usage: python crop.py <src> <dst> <x0> <y0> <x1> <y1> <scale>
src, dst = sys.argv[1], sys.argv[2]
x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7])
scale = float(sys.argv[7])
im = Image.open(src).convert("RGB")
w, h = im.size
x1, y1 = min(x1, w), min(y1, h)
c = im.crop((x0, y0, x1, y1))
if scale != 1.0:
    c = c.resize((max(1, int(c.width * scale)), max(1, int(c.height * scale))), Image.LANCZOS)
c.save(dst)
print("saved", dst, c.size)
