"""std bbox 归一化: 两种目标对比 + 用 IoU(std, GT) 量化对齐改善。

A 方案: std bbox -> 填满 256x256          (用户原话)
B 方案: std bbox -> 对齐到**该样本 GT 的 bbox** (逐样本, 对齐最优)
     GT bbox 用 2%~98% 分位, 抗杂点
"""
import csv, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
random.seed(21)


def bbox(m, lo=0, hi=100):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    if lo > 0 or hi < 100:
        x0, x1 = np.percentile(xs, [lo, hi])
        y0, y1 = np.percentile(ys, [lo, hi])
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)


def place(crop, box):
    x0, y0, x1, y1 = box
    tw, th = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
    out = np.zeros((256, 256), bool)
    a = np.asarray(im) > 127
    h = min(th, 256 - y0); w = min(tw, 256 - x0)
    out[y0:y0 + h, x0:x0 + w] = a[:h, :w]
    return out


def normA(s):
    b = bbox(s)
    if b is None:
        return s
    return place(s[b[1]:b[3] + 1, b[0]:b[2] + 1], (0, 0, 255, 255))


def normB(s, g):
    bs, bg = bbox(s), bbox(g, 2, 98)
    if bs is None or bg is None:
        return s
    return place(s[bs[1]:bs[3] + 1, bs[0]:bs[2] + 1], bg)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
samp = random.sample(rows, 800)
res = {"原样": [], "A填满256": [], "B对齐GT": []}
for r in samp:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    if not g.any() or not s.any():
        continue
    res["原样"].append(iou(s, g))
    res["A填满256"].append(iou(normA(s), g))
    res["B对齐GT"].append(iou(normB(s, g), g))

print(f"n={len(res['原样'])}   std 与 GT 墨迹的 IoU (越高=对齐越好)")
for k, v in res.items():
    v = np.array(v)
    print(f"   {k:<10} med={np.median(v):.3f}  mean={v.mean():.3f}  "
          f"p90={np.percentile(v,90):.3f}")

# ---- 海报 ----
sel = random.sample(samp, 4)
CELL, LBL = 256, 40
cv = Image.new("RGB", (4 * CELL + 5 * 10, len(sel) * (CELL + LBL) + 5 * 10), (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype("tools/fonts/simhei.ttf", 15)
TITLES = ["GT (真迹)", "std 原样 (红)", "A: std 填满 256", "B: std 对齐 GT bbox"]
for i, r in enumerate(sel):
    y0 = 10 + i * (CELL + LBL + 10)
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    sA, sB = normA(s), normB(s, g)
    panels = []
    panels.append(np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1))
    for sv in (s, sA, sB):
        ov = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
        ov[sv] = [255, 40, 40]
        panels.append(ov)
    for ci, im in enumerate(panels):
        x0 = 10 + ci * (CELL + 10)
        cv.paste(Image.fromarray(im), (x0, y0 + LBL))
        if i == 0:
            dr.text((x0 + 4, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
    dr.text((10, y0 + 22),
            "{}/{}  IoU: 原样 {:.3f} | A {:.3f} | B {:.3f}".format(
                r["calligrapher"], r["character"], iou(s, g), iou(sA, g), iou(sB, g)),
            font=fb, fill=(255, 225, 140))
cv.save("assets/std_norm_cmp.png")
print("-> assets/std_norm_cmp.png")
