"""A' 修正版: **保持长宽比**, 让长边填满 256 (scale = 256/max(w,h)), 居中。

对比:
  原样        std 不动
  A旧(错)     拉伸到 256x256 —— **压坏了长宽比** (用户指出的 bug)
  A'(新)      保持比例, 长边=256
  B           逐样本对齐到 GT bbox (推理时不可复现, 仅作上界参考)
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
        x0, x1 = np.percentile(xs, [lo, hi]); y0, y1 = np.percentile(ys, [lo, hi])
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)


def place(crop, tw, th, cx=128, cy=128):
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


def crop_std(s):
    b = bbox(s)
    return s[b[1]:b[3] + 1, b[0]:b[2] + 1], b


def normA_old(s):        # 错: 拉伸
    c, _ = crop_std(s)
    return place(c, 256, 256)


def normA_new(s):        # 对: 保持比例, 长边=256
    c, _ = crop_std(s)
    h, w = c.shape
    sc = 256.0 / max(h, w)
    return place(c, max(int(round(w * sc)), 1), max(int(round(h * sc)), 1))


def normB(s, g):
    c, _ = crop_std(s)
    bg = bbox(g, 2, 98)
    return place(c, bg[2] - bg[0] + 1, bg[3] - bg[1] + 1,
                 (bg[0] + bg[2]) / 2, (bg[1] + bg[3]) / 2)


def hit(s, g):
    return float((s & g).sum() / max(s.sum(), 1))


rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
samp = random.sample(rows, 800)
res = {"原样": [], "A旧(拉伸,错)": [], "A'(保比例)": [], "B(对齐GT,参考)": []}
for r in samp:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    if not g.any() or not s.any():
        continue
    res["原样"].append(hit(s, g))
    res["A旧(拉伸,错)"].append(hit(normA_old(s), g))
    res["A'(保比例)"].append(hit(normA_new(s), g))
    res["B(对齐GT,参考)"].append(hit(normB(s, g), g))

print(f"n={len(res['原样'])}   **骨架落墨率**")
for k, v in res.items():
    v = np.array(v)
    print(f"   {k:<16} med={np.median(v):.3f}  mean={v.mean():.3f}  "
          f"p10={np.percentile(v,10):.3f}")

sel = random.sample(samp, 4)
CELL, LBL = 256, 40
cv = Image.new("RGB", (5 * CELL + 6 * 8, len(sel) * (CELL + LBL) + 5 * 8), (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype("tools/fonts/simhei.ttf", 14)
TITLES = ["GT", "std 原样", "A旧 拉伸(错)", "A' 保比例长边=256", "B 对齐GT"]
for i, r in enumerate(sel):
    y0 = 8 + i * (CELL + LBL + 8)
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    vs = [None, s, normA_old(s), normA_new(s), normB(s, g)]
    for ci, sv in enumerate(vs):
        x0 = 8 + ci * (CELL + 8)
        im = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
        if sv is not None:
            im[sv] = [255, 40, 40]
        cv.paste(Image.fromarray(im), (x0, y0 + LBL))
        if i == 0:
            dr.text((x0 + 3, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
    dr.text((10, y0 + 22),
            "{}/{}  落墨率 原 {:.2f} | A旧 {:.2f} | A' {:.2f} | B {:.2f}".format(
                r["calligrapher"], r["character"], hit(s, g),
                hit(normA_old(s), g), hit(normA_new(s), g), hit(normB(s, g), g)),
            font=fb, fill=(255, 225, 140))
cv.save("assets/std_norm_v2.png")
print("-> assets/std_norm_v2.png")
