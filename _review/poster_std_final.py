"""归一化后的 std 叠在 GT 上 (最终确认)。"""
import csv, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
random.seed(33)
rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))


def bbox(m):
    ys, xs = np.where(m)
    return None if len(ys) == 0 else (int(xs.min()), int(ys.min()),
                                      int(xs.max()), int(ys.max()))


sel = random.sample(rows, 8)
CELL, LBL, NC = 256, 38, 4
NR = (len(sel) + NC - 1) // NC
cv = Image.new("RGB", (NC * CELL + (NC + 1) * 8, NR * (CELL + LBL) + (NR + 1) * 8),
               (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype("tools/fonts/simhei.ttf", 13)
for i, r in enumerate(sel):
    rr, cc = i // NC, i % NC
    x0, y0 = 8 + cc * (CELL + 8), 8 + rr * (CELL + LBL + 8)
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    im = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
    im[s] = [255, 40, 40]
    cv.paste(Image.fromarray(im), (x0, y0 + LBL))
    bg, bs = bbox(g), bbox(s)
    lg = max(bg[2] - bg[0], bg[3] - bg[1]) + 1
    ls = max(bs[2] - bs[0], bs[3] - bs[1]) + 1
    dr.text((x0 + 3, y0 + 3),
            "{} / {}".format(r["calligrapher"], r["character"]), font=fb,
            fill=(255, 225, 140))
    dr.text((x0 + 3, y0 + 21),
            "GT长边{} std长边{}".format(lg, ls), font=fb, fill=(150, 200, 255))
cv.save("assets/std_norm_final.png")
print("-> assets/std_norm_final.png")
