"""重渲染 strict 那条（044447）的 std g —— 从「升」改成「陞」。

## 用哪个字体
与 tools/render_missing_std.py 一致：STKAITI（楷体，最接近书法）。
渲染流程也一致：渲染 -> 二值 -> 骨架 -> 裁包围盒 -> 居中。
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation

os.chdir("/root/Workspace/xy/DiT")

SIZE = 256
FONT = "_fonts/STKAITI.TTF"
CH = "\u965e"          # 陞
OUT = "data/50k/std/044447.png"


def render_skeleton(ch, font_path, size=SIZE):
    fp = int(size * 0.82)
    font = ImageFont.truetype(font_path, fp)
    canvas = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(canvas)
    bbox = d.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]), ch,
           font=font, fill=0)
    arr = np.asarray(canvas)
    ink = arr < 127
    sk = skeletonize(ink)
    # ⚠ skimage 的 binary_dilation 在这个版本里既不吃 iterations 也不吃 num_iter
    #   -> 改用 scipy.ndimage.binary_dilation（API 稳定，8 邻域膨胀一次）
    sk = binary_dilation(sk, iterations=1)
    ys, xs = np.where(sk)
    if len(ys) == 0:
        raise RuntimeError("渲染为空")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = sk[y0:y1, x0:x1]
    hh, ww = crop.shape
    s = max(hh, ww)
    canvas2 = np.full((size, size), 255, np.uint8)
    oy, ox = (size - hh) // 2, (size - ww) // 2
    canvas2[oy:oy + hh, ox:ox + ww] = np.where(crop, 0, 255)
    return Image.fromarray(canvas2)


print(f"  渲染 {CH} (U+{ord(CH):04X}) 用 {FONT}")
img = render_skeleton(CH, FONT)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT)
print(f"  ✓ -> {OUT}")

# 复验
a = np.asarray(Image.open(OUT))
print(f"    尺寸={a.shape}  墨迹占比={float((a < 127).mean()):.4f}")

# 顺便对比：旧的是「升」
print(f"\n  对比参考:")
print(f"    训练集里「陞」的 std: data/50k/std/019962.png  "
      f"存在={os.path.exists('data/50k/std/019962.png')}")
