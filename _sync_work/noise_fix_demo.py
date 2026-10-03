"""噪点图最差样本海报 + 修法效果评估。

修法候选:
  F1 连通块过滤: 丢掉 < max(20px, 0.02*最大块) 的块
  F2 形态学开运算 3x3
  F3 F1 + F2
关键: 修法**不能伤干净图** —— 同时报告"干净图被削掉多少墨"。
"""
import csv, os, random, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as cc_label, median_filter, binary_opening, binary_closing

os.chdir("/root/Workspace/xy/DiT")
random.seed(3)
FONT = "tools/fonts/simhei.ttf"

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))


def load(p):
    return np.asarray(Image.open(p).convert("L").resize((256, 256), Image.LANCZOS)) < 128


def score(g):
    lab, nc = cc_label(g)
    sizes = np.bincount(lab.ravel())[1:]
    small = sizes < 20
    ink = max(g.sum(), 1)
    n1 = sizes[small].sum() / ink
    med = median_filter(g.astype(np.uint8), size=3) > 0
    n2 = float((g ^ med).mean())
    return n1, n2


def fix(g, min_cc=20, frac=0.02, opening=False):
    out = g.copy()
    if opening:
        out = binary_opening(out, structure=np.ones((3, 3), bool))
    lab, nc = cc_label(out)
    if nc:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        thr = max(min_cc, frac * sizes.max())
        keep = np.where(sizes >= thr)[0]
        out = np.isin(lab, keep)
    return out


# ---- 找最差样本 ----
rec = []
for r in w:
    try:
        g = load(r["image_path"])
    except Exception:
        continue
    if not g.any():
        continue
    n1, n2 = score(g)
    rec.append((n1 * 2 + n2 * 8, n1, n2, r))
rec.sort(key=lambda x: -x[0])
print("最差 12 个样本:")
print(f"{'书家':<8}{'字':<4}{'N1':>8}{'N2':>9}{'修后N1':>9}{'修后N2':>9}{'去掉墨%':>9}")
worst = rec[:12]
rows_draw = []
for s, n1, n2, r in worst:
    g = load(r["image_path"])
    f = fix(g, opening=True)
    f1, f2 = score(f)
    rm = 1 - f.sum() / max(g.sum(), 1)
    rows_draw.append((g, f, r, n1, n2, f1, f2, rm))
    print(f"{r['calligrapher']:<8}{r['character']:<4}{n1:>8.4f}{n2:>9.4f}"
          f"{f1:>9.4f}{f2:>9.4f}{rm*100:>8.1f}%")

# ---- 干净图会被削掉多少? ----
clean_rm = []
for r in random.sample(o, 300):          # 老数据基本干净
    g = load(r["image_path"])
    if not g.any():
        continue
    f = fix(g, opening=True)
    clean_rm.append(1 - f.sum() / max(g.sum(), 1))
print(f"\n干净图(老数据抽 300) 被削掉的墨: med={np.median(clean_rm)*100:.2f}%  "
      f"p90={np.percentile(clean_rm,90)*100:.2f}%  max={max(clean_rm)*100:.2f}%")

# ---- 海报 ----
CELL, LBL = 256, 46
ncol = 2
nrow = (len(rows_draw) + ncol - 1) // ncol
GAP = 10
W = ncol * (2 * CELL + GAP) + GAP
H = nrow * (CELL + LBL + GAP) + GAP
cv = Image.new("RGB", (W, H), (24, 24, 28))
dr = ImageDraw.Draw(cv)
try:
    f_big = ImageFont.truetype(FONT, 16)
    f_sm = ImageFont.truetype(FONT, 13)
except Exception:
    f_big = f_sm = ImageFont.load_default()
for i, (g, f, r, n1, n2, f1, f2, rm) in enumerate(rows_draw):
    rr, cc = i // ncol, i % ncol
    x0 = GAP + cc * (2 * CELL + GAP)
    y0 = GAP + rr * (CELL + LBL + GAP)
    dr.text((x0 + 4, y0 + 3),
            f"{r['calligrapher']} / {r['character']}   N1 {n1:.3f}->{f1:.3f}  "
            f"N2 {n2:.3f}->{f2:.3f}  去墨 {rm*100:.0f}%",
            font=f_big, fill=(255, 225, 140))
    for ci, (img, ttl) in enumerate([(g, "原始 GT"), (f, "修复后")]):
        arr = np.stack([img.astype(np.uint8) * 255] * 3, -1)
        cv.paste(Image.fromarray(arr), (x0 + ci * CELL, y0 + LBL))
        dr.text((x0 + ci * CELL + 6, y0 + 28), ttl, font=f_sm, fill=(170, 170, 180))
out = "assets/noise_worst12.png"
cv.save(out)
print(f"\nwritten {out}  {cv.size}")
