"""渲染"墨太重/糊团"样本海报。"""
import csv, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
paths = set(open("_sync_work/too_ink.txt", encoding="utf-8").read().split("\n"))
sel = [r for r in rows if r["image_path"] in paths][:8]

CELL, LBL, NC = 256, 40, 4
NR = (len(sel) + NC - 1) // NC
cv = Image.new("RGB", (NC * CELL + (NC + 1) * 8, NR * (CELL + LBL) + (NR + 1) * 8),
               (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype("tools/fonts/simhei.ttf", 14)
for i, r in enumerate(sel):
    rr, cc = i // NC, i % NC
    x0, y0 = 8 + cc * (CELL + 8), 8 + rr * (CELL + LBL + 8)
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    arr = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
    cv.paste(Image.fromarray(arr), (x0, y0 + LBL))
    lab = "{}/{}  ink={:.3f}".format(r["calligrapher"], r["character"], g.mean())
    dr.text((x0 + 4, y0 + 3), lab, font=fb, fill=(255, 225, 140))
cv.save("assets/50k_too_ink.png")
print("written assets/50k_too_ink.png", len(sel), "张")
