from PIL import Image
import numpy as np
import csv
from pathlib import Path

# Check all 34 common samples between Top2, Top5, Top10
r2 = list(csv.DictReader(open("exp-std/csv/eval_top2_100.csv", encoding="utf-8-sig")))
r5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))
r10 = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

img_to_r5 = {r["image_path"]: (i, r) for i, r in enumerate(r5)}
img_to_r10 = {r["image_path"]: (i, r) for i, r in enumerate(r10)}

common = []
for i2, r in enumerate(r2):
    p = r["image_path"]
    if p in img_to_r5 and p in img_to_r10:
        i5 = img_to_r5[p][0]
        i10 = img_to_r10[p][0]
        
        im30 = np.array(Image.open(f"assets/results/top2_u100_base_30k/g{i2}.png").convert("L"))
        im50 = np.array(Image.open(f"assets/results/top2_u100_base_50k/g{i2}.png").convert("L"))
        ink30 = int((im30 < 128).sum())
        ink50 = int((im50 < 128).sum())
        
        common.append({
            "i2": i2, "i5": i5, "i10": i10,
            "callig": r["calligrapher"], "script": r["script"], "char": r["character"],
            "ink30": ink30, "ink50": ink50,
            "diff": ink50 - ink30
        })

print(f"{'Sample':20s} | {'Char':4s} | {'30k Ink':8s} | {'50k Ink':8s} | {'Collapse Status'}")
print("-" * 65)
for c in common:
    info = f"i2={c['i2']:02d} i10={c['i10']:03d} {c['callig']} {c['script']}"
    if c['ink30'] > 2000 and c['ink50'] < 1600:
        status = "CRITICAL COLLAPSE (WHITE)"
    elif c['ink30'] < 500 and c['ink50'] < 500:
        status = "Early Failed (<30k)"
    elif c['ink50'] < c['ink30'] * 0.6:
        status = "Fading / Deteriorating"
    else:
        status = "Stable / Normal"
    print(f"{info:26s} | {c['char']:2s}   | {c['ink30']:7d}  | {c['ink50']:7d}  | {status}")
