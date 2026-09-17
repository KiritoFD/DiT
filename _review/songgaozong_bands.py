"""宋高宗按质量档位抽样目检 (极性已修正: 墨=黑, 底=白)。"""
import csv, os, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_opening

os.chdir("/root/Workspace/xy/DiT")
FONT = "tools/fonts/simhei.ttf"
STRUCT = np.ones((3, 3), bool)
TARGET = "宋高宗"

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
rows = [r for r in o + w if r["calligrapher"] == TARGET]

buckets = collections.defaultdict(list)
for r in rows:
    g = np.asarray(Image.open(r["image_path"]).convert("L")
                   .resize((256, 256), Image.LANCZOS)) < 128
    ink = max(g.sum(), 1)
    keep = binary_opening(g, structure=STRUCT).sum() / ink
    if ink < 10:
        buckets["empty"].append(r)
    elif keep < 0.05:
        buckets["pure_noise"].append(r)
    elif keep < 0.30:
        buckets["heavy"].append(r)
    elif keep < 0.60:
        buckets["light"].append(r)
    else:
        buckets["clean"].append(r)

print(f"{TARGET} 共 {len(rows)} 张:")
for k in ["clean", "light", "heavy", "pure_noise", "empty"]:
    n = len(buckets[k])
    print(f"   {k:<12} {n:>4}  {100*n/len(rows):>6.2f}%")

ORDER = ["clean", "light", "heavy", "pure_noise"]
PER = 4
CELL, LBL = 256, 42
W = PER * CELL + (PER + 1) * 10
H = len(ORDER) * (CELL + LBL) + (len(ORDER) + 1) * 10
cv = Image.new("RGB", (W, H), (24, 24, 28))
dr = ImageDraw.Draw(cv)
try:
    fb = ImageFont.truetype(FONT, 15)
except Exception:
    fb = ImageFont.load_default()

for ri, band in enumerate(ORDER):
    y0 = 10 + ri * (CELL + LBL + 10)
    dr.text((14, y0 + 8), f"--- {band} ({len(buckets[band])} 张) ---",
            font=fb, fill=(255, 120, 120))
    for ci, r in enumerate(buckets[band][:PER]):
        g = np.asarray(Image.open(r["image_path"]).convert("L")
                       .resize((CELL, CELL), Image.LANCZOS)) < 128
        arr = np.where(g, 0, 255).astype(np.uint8)      # ★ 墨=黑, 底=白
        x0 = 10 + ci * (CELL + 10)
        cv.paste(Image.fromarray(np.stack([arr] * 3, -1)), (x0, y0 + LBL))
        ink = max(g.sum(), 1)
        keep = binary_opening(g, structure=STRUCT).sum() / ink
        dr.text((x0 + 4, y0 + 3), f"{r['character']}  keep={keep:.3f}",
                font=fb, fill=(255, 225, 140))
out = "assets/songgaozong_bands.png"
cv.save(out)
print(f"written {out}  {cv.size}")
