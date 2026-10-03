"""测各字体能否区分繁/简字形 —— 这决定"按 character 重渲染"能不能修好条件。

对若干繁简对(復/复, 應/应, 將/将, 圖/图, 縣/县):
  每个字体分别渲染两形 -> 若两幅图几乎相同 = 字体把繁体映射成了简体(不可用)。
输出差异度量 (0 = 完全相同) + 拼图供人眼确认。
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
PAIRS = [("復", "复"), ("應", "应"), ("將", "将"), ("圖", "图"), ("縣", "县")]
FONTS = []
for p in ("_fonts/STKAITI.TTF", "_fonts/STSONG.TTF", "_fonts/Deng.ttf", "_fonts/msyh.ttc",
          "/tmp/sync_stage/tools/fonts/STXINGKA.TTF", "_fonts/STLITI.TTF",
          "/usr/share/fonts/truetype/arphic/uming.ttc"):
    if os.path.exists(p):
        FONTS.append(p)
print(f"[fonts] 可用 {len(FONTS)}: {FONTS}")


def render(ch, fp, size=200):
    f = ImageFont.truetype(fp, size)
    im = Image.new("L", (size, size), 255)
    ImageDraw.Draw(im).text((size // 2, size // 2), ch, font=f, fill=0, anchor="mm")
    a = np.asarray(im, np.float32)
    return a, float((a < 128).mean())


print(f"{'font':<42}{'对':<10}{'差异':>8}{'墨1':>8}{'墨2':>8}")
rows_img = []
for fp in FONTS:
    for a_ch, b_ch in PAIRS:
        try:
            ia, ma = render(a_ch, fp)
            ib, mb = render(b_ch, fp)
        except Exception as e:                               # noqa: BLE001
            print(f"{os.path.basename(fp):<42}{a_ch}/{b_ch:<6}  ERR {e}")
            continue
        diff = float(np.abs(ia - ib).mean())
        flag = "  <-- 两形几乎相同(字体不区分)" if diff < 0.5 else ""
        print(f"{os.path.basename(fp):<42}{a_ch}/{b_ch:<6}{diff:>8.1f}{ma:>8.3f}{mb:>8.3f}{flag}")
        if len(rows_img) < 24:
            rows_img.append((ia, ib))

S = 100
n = len(rows_img)
canvas = Image.new("L", (S * 2, S * n), 255)
for j, (a, b) in enumerate(rows_img):
    canvas.paste(Image.fromarray(a.astype(np.uint8)).resize((S, S)), (0, j * S))
    canvas.paste(Image.fromarray(b.astype(np.uint8)).resize((S, S)), (S, j * S))
os.makedirs("_ot_scratch", exist_ok=True)
canvas.save("_ot_scratch/font_pairs.png")
print("[图] _ot_scratch/font_pairs.png (左列=繁体字形, 右列=简体字形, 逐字体逐对)")
