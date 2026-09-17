"""按 GT 长边中位(228.5) ~ 90% 定目标: std 长边 -> 230px, 保持比例, 居中。"""
import csv, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
random.seed(21)
TARGET = 230


def bbox(m):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def norm(s, target=TARGET):
    b = bbox(s)
    if b is None:
        return s
    c = s[b[1]:b[3] + 1, b[0]:b[2] + 1]
    h, w = c.shape
    sc = float(target) / max(h, w)
    tw, th = max(int(round(w * sc)), 1), max(int(round(h * sc)), 1)
    im = Image.fromarray((c * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
    a = np.asarray(im) > 127
    out = np.zeros((256, 256), bool)
    x0, y0 = int(round(128 - tw / 2)), int(round(128 - th / 2))
    out[y0:y0 + th, x0:x0 + tw] = a
    return out


rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
sel = random.sample(rows, 5)
CELL, LBL = 256, 40
cv = Image.new("RGB", (3 * CELL + 4 * 8, len(sel) * (CELL + LBL) + 5 * 8), (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype("tools/fonts/simhei.ttf", 14)
TITLES = ["GT 真迹", "std 原样 (偏小)", "std 长边->230 (保持比例)"]
for i, r in enumerate(sel):
    y0 = 8 + i * (CELL + LBL + 8)
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    sn = norm(s)
    for ci, sv in enumerate([None, s, sn]):
        x0 = 8 + ci * (CELL + 8)
        im = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
        if sv is not None:
            im[sv] = [255, 40, 40]
        cv.paste(Image.fromarray(im), (x0, y0 + LBL))
        if i == 0:
            dr.text((x0 + 3, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
    bgg, bss = bbox(g), bbox(norm(s))
    dr.text((10, y0 + 22),
            "{}/{}  GT长边={}  std长边 {}->{}".format(
                r["calligrapher"], r["character"],
                max(bgg[2] - bgg[0], bgg[3] - bgg[1]) + 1,
                max(bbox(s)[2] - bbox(s)[0], bbox(s)[3] - bbox(s)[1]) + 1,
                max(bss[2] - bss[0], bss[3] - bss[1]) + 1),
            font=fb, fill=(255, 225, 140))
cv.save("assets/std_norm_230.png")
print("-> assets/std_norm_230.png")

# 全量核对: 归一化后 std 长边 vs GT 长边
gl, sl2 = [], []
for r in random.sample(rows, 400):
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    bg, bs = bbox(g), bbox(norm(s))
    if bg and bs:
        gl.append(max(bg[2] - bg[0], bg[3] - bg[1]) + 1)
        sl2.append(max(bs[2] - bs[0], bs[3] - bs[1]) + 1)
print(f"   归一化后: std 长边 med={np.median(sl2):.1f}  GT 长边 med={np.median(gl):.1f}"
      f"  -> 比值 {np.median(sl2)/np.median(gl):.3f} (理想 1.0)")
