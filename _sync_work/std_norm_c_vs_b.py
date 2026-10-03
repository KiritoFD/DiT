"""C 方案: std bbox -> 固定目标尺度(GT 典型填充 0.875 -> 224px), 居中。
   与 A(填满256) / B(对齐GT bbox) 对比。
指标用 **骨架落墨率** (std 像素落在 GT 墨内的比例) —— 比 IoU 直接得多。
"""
import csv, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
random.seed(21)
TARGET = 224          # = 256 * 0.875 (GT 的典型填充率)


def bbox(m, lo=0, hi=100):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    if lo > 0 or hi < 100:
        x0, x1 = np.percentile(xs, [lo, hi]); y0, y1 = np.percentile(ys, [lo, hi])
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)


def place(crop, tw, th, cx, cy):
    """把 crop 缩到 (tw,th), 中心放在 (cx,cy)"""
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(tw, 1), max(th, 1)), Image.NEAREST)
    a = np.asarray(im) > 127
    out = np.zeros((256, 256), bool)
    x0 = int(round(cx - tw / 2)); y0 = int(round(cy - th / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    h = min(a.shape[0] - sy0, 256 - y0); w = min(a.shape[1] - sx0, 256 - x0)
    if h > 0 and w > 0:
        out[y0:y0 + h, x0:x0 + w] = a[sy0:sy0 + h, sx0:sx0 + w]
    return out


def normA(s):
    b = bbox(s)
    return place(s[b[1]:b[3] + 1, b[0]:b[2] + 1], 256, 256, 128, 128)


def normB(s, g):
    bs, bg = bbox(s), bbox(g, 2, 98)
    return place(s[bs[1]:bs[3] + 1, bs[0]:bs[2] + 1],
                 bg[2] - bg[0] + 1, bg[3] - bg[1] + 1,
                 (bg[0] + bg[2]) / 2, (bg[1] + bg[3]) / 2)


def normC(s):
    bs = bbox(s)
    w, h = bs[2] - bs[0] + 1, bs[3] - bs[1] + 1
    sc = TARGET / max(w, h)
    return place(s[bs[1]:bs[3] + 1, bs[0]:bs[2] + 1],
                 max(int(round(w * sc)), 1), max(int(round(h * sc)), 1), 128, 128)


def hitrate(s, g):
    return float((s & g).sum() / max(s.sum(), 1))


rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
samp = random.sample(rows, 800)
res = {"原样": [], "A填满256": [], "B对齐GT": [], "C固定224": []}
for r in samp:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    if not g.any() or not s.any():
        continue
    res["原样"].append(hitrate(s, g))
    res["A填满256"].append(hitrate(normA(s), g))
    res["B对齐GT"].append(hitrate(normB(s, g), g))
    res["C固定224"].append(hitrate(normC(s), g))

print(f"n={len(res['原样'])}   **骨架落墨率** (std 像素落在 GT 墨内的比例)")
for k, v in res.items():
    v = np.array(v)
    print(f"   {k:<10} med={np.median(v):.3f}  mean={v.mean():.3f}  "
          f"p10={np.percentile(v,10):.3f}")

sel = random.sample(samp, 4)
CELL, LBL = 256, 40
cv = Image.new("RGB", (5 * CELL + 6 * 8, len(sel) * (CELL + LBL) + 5 * 8), (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype("tools/fonts/simhei.ttf", 14)
TITLES = ["GT", "std 原样", "A 填满256", "B 对齐GT", "C 固定224"]
for i, r in enumerate(sel):
    y0 = 8 + i * (CELL + LBL + 8)
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    vs = [None, s, normA(s), normB(s, g), normC(s)]
    for ci, sv in enumerate(vs):
        x0 = 8 + ci * (CELL + 8)
        if sv is None:
            im = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
        else:
            im = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
            im[sv] = [255, 40, 40]
        cv.paste(Image.fromarray(im), (x0, y0 + LBL))
        if i == 0:
            dr.text((x0 + 3, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
    dr.text((10, y0 + 22),
            "{}/{}  落墨率 原样 {:.2f} | A {:.2f} | B {:.2f} | C {:.2f}".format(
                r["calligrapher"], r["character"], hitrate(s, g), hitrate(normA(s), g),
                hitrate(normB(s, g), g), hitrate(normC(s), g)),
            font=fb, fill=(255, 225, 140))
cv.save("assets/std_norm_abc.png")
print("-> assets/std_norm_abc.png")
