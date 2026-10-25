import csv
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import label

r2 = list(csv.DictReader(open("exp-std/csv/eval_top2_100.csv", encoding="utf-8-sig")))
r5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))
r10 = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

img_to_r5 = {r["image_path"]: (i, r) for i, r in enumerate(r5)}
img_to_r10 = {r["image_path"]: (i, r) for i, r in enumerate(r10)}

for i2, r in enumerate(r2):
    p = r["image_path"]
    if p in img_to_r5 and p in img_to_r10:
        i5 = img_to_r5[p][0]
        i10 = img_to_r10[p][0]
        
        im = Image.open(p).convert("L")
        arr = np.array(im)
        ink = arr < 128
        y, x = np.where(ink)
        ymin, ymax = y.min(), y.max()
        xmin, xmax = x.min(), x.max()
        
        lbl, num = label(ink)
        sizes = np.bincount(lbl.ravel())[1:] if num > 0 else []
        debris = sum(s < 40 for s in sizes)
        
        # Check border touches: does ink touch row 0 or row 255 or col 0 or col 255?
        touches_edge = (ymin == 0 or ymax == 255 or xmin == 0 or xmax == 255)
        
        c = r["calligrapher"]
        s = r["script"]
        ch = r["character"]
        print(f"i2={i2:02d} i5={i5:02d} i10={i10:03d} | {c} {s} [{ch}] | y=[{ymin:3d},{ymax:3d}], x=[{xmin:3d},{xmax:3d}], touches={touches_edge}, comp={num:3d}, debris={debris:3d}")
