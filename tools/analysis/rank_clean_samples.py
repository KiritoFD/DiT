import csv
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import label

# We have:
# 1) exp-std/csv/eval_top2_100.csv (100 samples)
# 2) exp-std/csv/eval_top5_fixed.csv (93 samples)
# 3) exp-std/csv/eval200_fixed.csv (187 samples)

r2 = list(csv.DictReader(open("exp-std/csv/eval_top2_100.csv", encoding="utf-8-sig")))
r5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))
r10 = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

img_to_idx_10 = {r["image_path"]: i for i, r in enumerate(r10)}
img_to_idx_5 = {r["image_path"]: i for i, r in enumerate(r5)}

print("=== Evaluating ALL 34 common samples ===")
samples = []
for i2, r in enumerate(r2):
    p = r["image_path"]
    if p in img_to_idx_5 and p in img_to_idx_10:
        i5 = img_to_idx_5[p]
        i10 = img_to_idx_10[p]
        
        im_gt = Image.open(p).convert("L")
        arr = np.array(im_gt)
        ink = arr < 128
        ink_sum = int(ink.sum())
        
        # Check margins: at least 15 pixels padding on all 4 sides!
        m_top = int(ink[:15, :].sum())
        m_bot = int(ink[-15:, :].sum())
        m_left = int(ink[:, :15].sum())
        m_right = int(ink[:, -15:].sum())
        
        # Check components and small debris
        lbl, num = label(ink)
        sizes = np.bincount(lbl.ravel())[1:] if num > 0 else []
        debris = int(sum(s < 40 for s in sizes))
        main_strokes = int(sum(s >= 40 for s in sizes))
        
        # Score quality:
        # Penalties:
        # - border touching: -1000
        # - debris count: -50 each
        # - too few/too many ink pixels: penalty
        score = 100
        if m_top > 0 or m_bot > 0 or m_left > 0 or m_right > 0:
            score -= 1000
        score -= debris * 20
        if num > 15:
            score -= (num - 15) * 10
        if ink_sum < 2500 or ink_sum > 25000:
            score -= 50
            
        samples.append({
            "i2": i2, "i5": i5, "i10": i10,
            "callig": r["calligrapher"], "script": r["script"], "char": r["character"],
            "ink": ink_sum, "num_comp": num, "debris": debris,
            "m_top": m_top, "m_bot": m_bot, "m_left": m_left, "m_right": m_right,
            "score": score,
            "path": p
        })

samples.sort(key=lambda x: x["score"], reverse=True)

print(f"{'Rank':4s} | {'Sample':18s} | {'Char':4s} | {'Score':6s} | {'Ink':5s} | {'Comp':4s} | {'Debris':6s} | {'Margins (T/B/L/R)':18s}")
print("-" * 80)
for rank, s in enumerate(samples):
    m_str = f"{s['m_top']}/{s['m_bot']}/{s['m_left']}/{s['m_right']}"
    info = f"{s['callig']} {s['script']}"
    print(f"#{rank+1:02d}  | i2={s['i2']:02d} i5={s['i5']:02d} i10={s['i10']:03d} {info:10s} | {s['char']:2s}   | {s['score']:6d} | {s['ink']:5d} | {s['num_comp']:4d} | {s['debris']:6d} | {m_str:18s}")
