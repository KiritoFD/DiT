import csv
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import label

# Let's inspect all 187 samples in eval200_fixed.csv
r10 = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))
r2 = list(csv.DictReader(open("exp-std/csv/eval_top2_100.csv", encoding="utf-8-sig")))
r5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))

img_to_idx_2 = {r["image_path"]: i for i, r in enumerate(r2)}
img_to_idx_5 = {r["image_path"]: i for i, r in enumerate(r5)}

print("Scanning for PERFECT, pristine samples...")

candidates = []

for idx, r in enumerate(r10):
    p_rel = r["image_path"]
    p_gt = Path(p_rel)
    if not p_gt.exists():
        continue
        
    im_gt = Image.open(p_gt).convert("L")
    arr_gt = np.array(im_gt)
    ink_gt = arr_gt < 128
    ink_sum = int(ink_gt.sum())
    
    # Strict border clearance: at least 15 pixels empty on ALL 4 sides!
    m_top = int(ink_gt[:15, :].sum())
    m_bot = int(ink_gt[-15:, :].sum())
    m_left = int(ink_gt[:, :15].sum())
    m_right = int(ink_gt[:, -15:].sum())
    if m_top > 0 or m_bot > 0 or m_left > 0 or m_right > 0:
        continue
        
    # Check GT fragmentation: NO broken debris!
    lbl, num = label(ink_gt)
    if num == 0:
        continue
    sizes = np.bincount(lbl.ravel())[1:]
    debris = int(sum(s < 40 for s in sizes))
    if debris > 2 or num > 12:
        continue
        
    # Check ink area
    if ink_sum < 2500 or ink_sum > 22000:
        continue
        
    in_top2 = p_rel in img_to_idx_2
    in_top5 = p_rel in img_to_idx_5
    
    candidates.append({
        "i10": idx,
        "i2": img_to_idx_2.get(p_rel, -1),
        "i5": img_to_idx_5.get(p_rel, -1),
        "callig": r["calligrapher"],
        "script": r["script"],
        "char": r["character"],
        "ink": ink_sum,
        "comp": num,
        "debris": debris,
        "in_all": in_top2 and in_top5,
        "path": p_rel
    })

print(f"Total pristine candidates in eval200_fixed: {len(candidates)}")
common_pristine = [c for c in candidates if c["in_all"]]
print(f"Candidates present in ALL THREE (Top2, Top5, Top10): {len(common_pristine)}")

for c in common_pristine:
    print(f"  Top10:#{c['i10']:03d} Top2:#{c['i2']:02d} Top5:#{c['i5']:02d} | {c['callig']} {c['script']} 【{c['char']}】 | ink={c['ink']}, comp={c['comp']}, debris={c['debris']}")
