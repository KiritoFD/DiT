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

print("=== Checking all 100 samples in Top2 ===")
clean_list = []

for i2, r in enumerate(r2):
    p = r["image_path"]
    p_path = Path(p)
    if not p_path.exists():
        continue
    im = Image.open(p_path).convert("L")
    arr = np.array(im)
    ink = arr < 128
    ink_sum = int(ink.sum())
    
    # 15px border clearance
    m_top = int(ink[:15, :].sum())
    m_bot = int(ink[-15:, :].sum())
    m_left = int(ink[:, :15].sum())
    m_right = int(ink[:, -15:].sum())
    touches = (m_top > 0 or m_bot > 0 or m_left > 0 or m_right > 0)
    
    lbl, num = label(ink)
    sizes = np.bincount(lbl.ravel())[1:] if num > 0 else []
    debris = int(sum(s < 40 for s in sizes))
    
    in_5 = p in img_to_r5
    in_10 = p in img_to_r10
    
    if not touches and num <= 12 and debris <= 2 and ink_sum >= 3500 and ink_sum <= 22000:
        clean_list.append({
            "i2": i2,
            "callig": r["calligrapher"],
            "script": r["script"],
            "char": r["character"],
            "ink": ink_sum,
            "comp": num,
            "debris": debris,
            "in_5": in_5,
            "i5": img_to_r5[p][0] if in_5 else -1,
            "in_10": in_10,
            "i10": img_to_r10[p][0] if in_10 else -1,
            "path": p
        })

print(f"Total clean samples in Top2: {len(clean_list)}")
for c in clean_list:
    print(f"i2={c['i2']:02d} | i5={c['i5']:02d} | i10={c['i10']:03d} | {c['callig']} {c['script']} [{c['char']}] | ink={c['ink']:5d}, comp={c['comp']:2d}, debris={c['debris']:1d} | in_5={c['in_5']}, in_10={c['in_10']}")
