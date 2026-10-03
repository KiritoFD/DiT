"""数据质量目检海报: 每行 = [GT 原图 | std 骨架(3px) | 叠加]  x 2 组/行。

抽样覆盖: 楷/行/隶 各 4 张, 老数据与 HCSU 混合, 含 top 书家与冷门书家。
标签带墨点率, 便于量化判断。
"""
import csv, os, collections, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
random.seed(7)
CELL = 256
LBL = 52
FONT = "tools/fonts/simhei.ttf"

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))
for r in o:
    r["_src"] = "old"
for r in w:
    r["_src"] = "hcsu"
m = o + w


def load_gt(r):
    im = Image.open(r["image_path"]).convert("L").resize((CELL, CELL), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def load_std(r):
    im = Image.open(r["std_path"]).convert("L").resize((CELL, CELL), Image.NEAREST)
    return np.asarray(im, dtype=np.uint8)


# ---- 抽样: 每书体 4 张, 兼顾 老/HCSU 与 top/冷门书家 ----
top = {c for c, _ in collections.Counter(r["calligrapher"] for r in m).most_common(10)}
sel = []
for sc in ["楷", "行", "隶"]:
    pool = [r for r in m if r["script"] == sc]
    # 1 张老+top书家, 1 张 HCSU+top书家, 1 张 HCSU+冷门, 1 张老+冷门
    for src, want_top in [("old", True), ("hcsu", True), ("hcsu", False), ("old", False)]:
        c = [r for r in pool if r["_src"] == src and ((r["calligrapher"] in top) == want_top)]
        if c:
            sel.append(random.choice(c))

# ---- 画布: 每行 2 组, 每组 3 格 ----
nrow = (len(sel) + 1) // 2
GAP = 10
W = 2 * (3 * CELL + GAP) + GAP
H = nrow * (CELL + LBL + GAP) + GAP
canvas = Image.new("RGB", (W, H), (24, 24, 28))
dr = ImageDraw.Draw(canvas)
try:
    f = ImageFont.truetype(FONT, 17)
    fs = ImageFont.truetype(FONT, 14)
except Exception:
    f = fs = ImageFont.load_default()

COL_TITLE = ["GT 真迹", "std 骨架 (3px)", "叠加 (红=std)"]
for gi, r in enumerate(sel):
    row, col = gi // 2, gi % 2
    x0 = GAP + col * (3 * CELL + GAP)
    y0 = GAP + row * (CELL + LBL + GAP)
    gt, st = load_gt(r), load_std(r)
    ov = np.stack([gt, gt, gt], -1)
    ink = st < 128
    ov[ink] = [255, 60, 60]
    cells = [np.stack([gt] * 3, -1), np.stack([st] * 3, -1), ov]
    for ci, c in enumerate(cells):
        canvas.paste(Image.fromarray(c.astype(np.uint8)),
                     (x0 + ci * CELL, y0 + LBL))
        dr.text((x0 + ci * CELL + 6, y0 + 30), COL_TITLE[ci], font=fs,
                fill=(170, 170, 180))
    ir_gt = float((gt < 128).mean())
    ir_st = float(ink.mean())
    dr.text((x0 + 4, y0 + 4),
            f"{r['calligrapher']} / {r['character']} / {r['script']} "
            f"[{r['_src']}]  gt_ink={ir_gt:.3f} std_ink={ir_st:.4f}",
            font=f, fill=(255, 225, 140))

out = "assets/data_quality_sample.png"
canvas.save(out)
print(f"written {out}  {canvas.size}  ({len(sel)} 组)")

# ---- 数值小结 ----
print()
print(f"{'书家':<10}{'字':<4}{'书体':<4}{'源':<6}{'GT墨点率':>10}{'std墨点率':>11}{'std/gt':>9}")
for r in sel:
    gt, st = load_gt(r), load_std(r)
    a = float((gt < 128).mean())
    b = float((st < 128).mean())
    print(f"{r['calligrapher']:<10}{r['character']:<4}{r['script']:<4}{r['_src']:<6}"
          f"{a:>10.3f}{b:>11.4f}{b/max(a,1e-9):>9.3f}")
