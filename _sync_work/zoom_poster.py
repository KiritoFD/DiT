"""把 calibration poster 放大, 便于肉眼判笔画 (分辨率太低会把细笔画看成 '空白')。"""
import os
import sys

from PIL import Image

src = sys.argv[1]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
im = Image.open(src).convert("L")
w, h = im.size
out = os.path.splitext(src)[0]
# 左右两半各自放大 (整幅放大后仍太宽, 看不清单字)
for nm, box in (("L", (0, 0, w // 2, h)), ("R", (w // 2, 0, w, h))):
    c = im.crop(box)
    c = c.resize((int(c.width * scale * 2), int(c.height * scale)), Image.LANCZOS)
    p = f"{out}_zoom{nm}.png"
    c.save(p)
    print(f"{p}  {c.size}")
