#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 train_top30_clean.csv 剔除 grey_ratio > 0.3 的脏图, 原地覆盖。"""
import os, sys, csv, json, shutil
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(BASE, "5script", "train_top30_clean.csv")
THRESH = 0.3

def fix_path(p):
    if p.startswith("final_images/"):
        return p.replace("final_images/", "final_imgs_256/", 1)
    return p

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
print(f"original: {len(rows)} rows")

# 计算每张图的 grey_ratio
def analyze_one(r):
    p = fix_path(r["image_path"])
    full = os.path.join(BASE, p)
    try:
        img = Image.open(full).convert("RGB").resize((256, 256), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        grey = ((arr > 0.15) & (arr < 0.85)).mean()
        return float(grey)
    except Exception:
        return 1.0  # 读不了的也删

print("analyzing...")
with ThreadPoolExecutor(max_workers=32) as pool:
    greys = list(pool.map(analyze_one, rows))

dirty_mask = np.array(greys) > THRESH
dirty_count = dirty_mask.sum()
print(f"dirty (grey>{THRESH}): {dirty_count} ({100*dirty_count/len(rows):.2f}%)")

# 备份
bak = CSV_PATH + ".bak_dirty"
shutil.copy2(CSV_PATH, bak)
print(f"backup: {bak}")

# 过滤
clean_rows = [r for r, d in zip(rows, dirty_mask) if not d]
print(f"after filter: {len(clean_rows)} rows (removed {len(rows) - len(clean_rows)})")

# 写回 (保留原列顺序)
with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for r in clean_rows:
        w.writerow(r)
print(f"wrote: {CSV_PATH}")

# 按 script 统计删减
from collections import Counter
removed_by_script = Counter(r["script"] for r, d in zip(rows, dirty_mask) if d)
kept_by_script = Counter(r["script"] for r, d in zip(rows, dirty_mask) if not d)
print("\n按 script:")
for s in sorted(removed_by_script.keys() | kept_by_script.keys()):
    rm = removed_by_script.get(s, 0)
    kp = kept_by_script.get(s, 0)
    print(f"  {s}: kept={kp}, removed={rm}")
