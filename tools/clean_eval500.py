#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""清洗 eval500: 删除 grey_ratio > 0.2 的图, 写回 eval500_clean.csv。"""
import csv, os, sys, shutil, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
CSV = os.path.join(BASE, "5script", "eval500_clean.csv")
THRESH = 0.2

def fix_path(p):
    if p.startswith("final_images/"):
        return p.replace("final_images/", "final_imgs_256/", 1)
    return p

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"original: {len(rows)} rows")

greys = []
for r in rows:
    p = os.path.join(BASE, fix_path(r["image_path"]))
    try:
        img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        grey = float(((arr > 0.15) & (arr < 0.85)).mean())
    except Exception:
        grey = 1.0
    greys.append(grey)

dirty = sum(1 for g in greys if g > THRESH)
print(f"dirty (grey>{THRESH}): {dirty} ({100*dirty/len(rows):.1f}%)")

bak = CSV + ".bak_dirty"
shutil.copy2(CSV, bak)
print(f"backup: {bak}")

clean_rows = [r for r, g in zip(rows, greys) if g <= THRESH]
print(f"after: {len(clean_rows)} rows (removed {len(rows)-len(clean_rows)})")

with open(CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for r in clean_rows:
        w.writerow(r)
print(f"wrote: {CSV}")

from collections import Counter
sc = Counter(r["script"] for r in clean_rows)
print(f"script dist: {dict(sc)}")
