"""验证"坏图"判据: 把高坏图率书家里被判坏的图渲出来目检。

⚠ 极性: 真实 GT 是**白底黑字**; 之前海报我画反了 (ink->白), 看着像反色。
   本脚本: ink=False(背景) -> 255 白, ink=True(墨) -> 0 黑。
"""
import csv, os, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as cc_label, binary_opening

os.chdir("/root/Workspace/xy/DiT")
FONT = "tools/fonts/simhei.ttf"
STRUCT = np.ones((3, 3), bool)

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
for r in o:
    r["_src"] = "old"
for r in w:
    r["_src"] = "hcsu"
m = o + w

bad_paths = set(open("_sync_work/bad_images.txt", encoding="utf-8").read().split("\n"))
TARGETS = ["宋高宗", "薛稷", "钟繇", "王献之", "褚遂良", "王羲之", "黄庭坚", "赵佶"]
PER = 3

CELL, LBL = 256, 46
ncol = PER
nrow = len(TARGETS)
GAP = 10
W = ncol * CELL + (ncol + 1) * GAP
H = nrow * (CELL + LBL) + (nrow + 1) * GAP
cv = Image.new("RGB", (W, H), (24, 24, 28))
dr = ImageDraw.Draw(cv)
try:
    fb = ImageFont.truetype(FONT, 15)
except Exception:
    fb = ImageFont.load_default()

for ri, c in enumerate(TARGETS):
    cand = [r for r in m if r["calligrapher"] == c and r["image_path"] in bad_paths]
    y0 = GAP + ri * (CELL + LBL + GAP)
    for ci, r in enumerate(cand[:PER]):
        g = np.asarray(Image.open(r["image_path"]).convert("L")
                       .resize((CELL, CELL), Image.LANCZOS)) < 128
        ink = max(g.sum(), 1)
        keep = binary_opening(g, structure=STRUCT).sum() / ink
        lab, nc = cc_label(g)
        sizes = np.bincount(lab.ravel())[1:]
        lcf = sizes.max() / ink if len(sizes) else 0
        # ★ 极性: 墨->黑(0), 背景->白(255)
        arr = np.where(g, 0, 255).astype(np.uint8)
        x0 = GAP + ci * (CELL + GAP)
        cv.paste(Image.fromarray(np.stack([arr] * 3, -1)), (x0, y0 + LBL))
        dr.text((x0 + 4, y0 + 3),
                f"{c} / {r['character']} [{r['_src']}]",
                font=fb, fill=(255, 225, 140))
        dr.text((x0 + 4, y0 + 24),
                f"keep={keep:.2f} lcf={lcf:.2f} ink={ink/g.size:.3f}",
                font=fb, fill=(160, 200, 255))
out = "assets/bad_validate.png"
cv.save(out)
print(f"written {out}  {cv.size}")
