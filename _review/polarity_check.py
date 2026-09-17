"""极性判定: 把 std 骨架(红) 叠在 GT 上。
骨架是 skeletonize(GT<128) 来的 —— 它必然落在**墨**上。
  - 若红线落在**黑**笔画上  -> 黑=墨, 正常
  - 若红线落在**白**区域上  -> 白=墨, 该图反色
同时给"原样/反色"并排, 人眼最终裁决。
"""
import csv, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
paths = set(open("_sync_work/too_ink.txt", encoding="utf-8").read().split("\n"))
sel = [r for r in rows if r["image_path"] in paths][:4]

CELL, LBL = 256, 44
cv = Image.new("RGB", (3 * CELL + 4 * 10, len(sel) * (CELL + LBL) + 5 * 10), (24, 24, 28))
dr = ImageDraw.Draw(cv)
fb = ImageFont.truetype("tools/fonts/simhei.ttf", 15)
TITLES = ["GT (原样)", "std 骨架叠 GT (红=骨架)", "GT 反色"]

for i, r in enumerate(sel):
    y0 = 10 + i * (CELL + LBL + 10)
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    ov = np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1)
    ov[s] = [255, 40, 40]
    inv = np.stack([np.where(g, 255, 0).astype(np.uint8)] * 3, -1)
    for ci, im in enumerate([np.stack([np.where(g, 0, 255).astype(np.uint8)] * 3, -1),
                             ov, inv]):
        x0 = 10 + ci * (CELL + 10)
        cv.paste(Image.fromarray(im), (x0, y0 + LBL))
        if i == 0:
            dr.text((x0 + 4, y0 + 3), TITLES[ci], font=fb, fill=(160, 200, 255))
    # 定量: 骨架落在"暗"像素上的比例
    hit = (s & g).sum() / max(s.sum(), 1)
    dr.text((10, y0 + 26), "{}/{}  骨架落在暗像素的比例={:.3f}".format(
        r["calligrapher"], r["character"], hit), font=fb, fill=(255, 225, 140))

cv.save("assets/polarity_check.png")
print("written assets/polarity_check.png")
for r in sel:
    g = np.asarray(Image.open(r["image_path"]).convert("L")) < 128
    s = np.asarray(Image.open(r["std_path"]).convert("L")) < 128
    print("  {}/{}: 骨架落暗率={:.3f}".format(r["calligrapher"], r["character"],
                                              (s & g).sum() / max(s.sum(), 1)))
