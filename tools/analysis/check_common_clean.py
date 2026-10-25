import csv
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import label

r2 = list(csv.DictReader(open("exp-std/csv/eval_top2_100.csv", encoding="utf-8-sig")))
r5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))
r10 = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

img_to_idx_10 = {r["image_path"]: i for i, r in enumerate(r10)}
img_to_idx_5 = {r["image_path"]: i for i, r in enumerate(r5)}

clean_common = []
for i2, r in enumerate(r2):
    p = r["image_path"]
    if p in img_to_idx_5 and p in img_to_idx_10:
        i5 = img_to_idx_5[p]
        i10 = img_to_idx_10[p]
        
        im = Image.open(p).convert("L")
        arr = np.array(im)
        ink = arr < 128
        ink_sum = int(ink.sum())
        
        m_top = int(ink[:12, :].sum())
        m_bottom = int(ink[-12:, :].sum())
        m_left = int(ink[:, :12].sum())
        m_right = int(ink[:, -12:].sum())
        touches = bool(m_top > 0 or m_bottom > 0 or m_left > 0 or m_right > 0)
        
        lbl, num = label(ink)
        sizes = np.bincount(lbl.ravel())[1:] if num > 0 else []
        small_pieces = int(sum(s < 35 for s in sizes))
        
        is_clean = (not touches and num <= 14 and small_pieces <= 3 and ink_sum > 2000)
        c = r["calligrapher"]
        s = r["script"]
        ch = r["character"]
        tag = "CLEAN" if is_clean else "DIRTY"
        print(f"i2={i2:02d} i5={i5:02d} i10={i10:03d} | {c} {s} [{ch}] | ink={ink_sum:5d}, comp={num:2d}, small={small_pieces:2d}, touches={touches} -> {tag}")
