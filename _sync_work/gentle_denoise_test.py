"""温和去噪对比: 3x3 中值 vs 3x3 开运算, 在 宋高宗 各档上试, 并测对干净图的伤害。"""
import csv, os, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_opening, median_filter

os.chdir("/root/Workspace/xy/DiT")
FONT = "tools/fonts/simhei.ttf"
ST = np.ones((3, 3), bool)


def load(p):
    return np.asarray(Image.open(p).convert("L").resize((256, 256), Image.LANCZOS)) < 128


def med(g):
    return median_filter(g.astype(np.uint8), size=3) > 0


def opn(g):
    return binary_opening(g, structure=ST)


def keep_ratio(a, b):
    return b.sum() / max(a.sum(), 1)


o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
sg = [r for r in o + w if r["calligrapher"] == "宋高宗"]

buckets = collections.defaultdict(list)
for r in sg:
    g = load(r["image_path"])
    k = keep_ratio(g, opn(g))
    b = "pure" if k < 0.05 else ("heavy" if k < 0.30 else ("light" if k < 0.60 else "clean"))
    buckets[b].append((r, g, k))

print(f"{'档':<8}{'n':>5}{'中值后保留':>12}{'开运算后保留':>14}")
for b in ["clean", "light", "heavy", "pure"]:
    if not buckets[b]:
        continue
    mr = [keep_ratio(g, med(g)) for _, g, _ in buckets[b]]
    orr = [keep_ratio(g, opn(g)) for _, g, _ in buckets[b]]
    print(f"{b:<8}{len(buckets[b]):>5}{np.mean(mr):>12.3f}{np.mean(orr):>14.3f}")

# 对"干净图"的伤害 (老数据, 无纹理)
import random
random.seed(5)
cl = [load(r["image_path"]) for r in random.sample(o, 200)]
print(f"\n干净图(老抽 200) 保留率: 中值 {np.mean([keep_ratio(g, med(g)) for g in cl]):.3f}"
      f"   开运算 {np.mean([keep_ratio(g, opn(g)) for g in cl]):.3f}")

# ---- 海报: light/heavy 各 2 张, 列 = [原 | 中值 | 开运算] ----
rows_draw = buckets["light"][:2] + buckets["heavy"][:2]
CELL, LBL = 256, 42
W = 3 * CELL + 4 * 10
H = len(rows_draw) * (CELL + LBL) + (len(rows_draw) + 1) * 10
cv = Image.new("RGB", (W, H), (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype(FONT, 15)
TITLES = ["原始", "3x3 中值", "3x3 开运算"]
for ri, (r, g, k) in enumerate(rows_draw):
    y0 = 10 + ri * (CELL + LBL + 10)
    imgs = [g, med(g), opn(g)]
    for ci, im in enumerate(imgs):
        x0 = 10 + ci * (CELL + 10)
        arr = np.where(im, 0, 255).astype(np.uint8)
        cv.paste(Image.fromarray(np.stack([arr] * 3, -1)), (x0, y0 + LBL))
        if ri == 0:
            dr.text((x0 + 4, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
    dr.text((14, y0 + 24), f"{r['character']}  keep={k:.3f}", font=fb, fill=(255, 225, 140))
out = "assets/denoise_cmp.png"
cv.save(out)
print(f"written {out}  {cv.size}")
