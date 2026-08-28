# -*- coding: utf-8 -*-
"""Quick check: image_path patterns + file existence for common csv."""
import os, sys, csv
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(BASE, "5script", "train_3top30_common.csv"), encoding="utf-8")))
print("rows:", len(rows))
print("first 3 image_path:")
for r in rows[:3]:
    print("  ", r["image_path"])
# pattern types
import re, collections
pat = collections.Counter()
for r in rows:
    p = r["image_path"]
    if p.startswith("final_imgs_256/"):
        pat["final_imgs_256"] += 1
    elif p.startswith("final_images/"):
        pat["final_images"] += 1
    else:
        pat["other"] += 1
print("path patterns:", dict(pat))
# check existence on a few
import glob
for r in rows[:5]:
    p = r["image_path"]
    full = os.path.join(BASE, p) if not os.path.isabs(p) else p
    # the path is relative to repo root; check
    print(" ", p, "exists:", os.path.exists(full))
# also try with final_imgs_256 dir (most likely)
print("first image via final_imgs_256:")
m = re.search(r"(\d+)\.png", rows[0]["image_path"])
if m:
    iid = int(m.group(1))
    p1 = os.path.join(BASE, "final_imgs_256", f"{iid}.png")
    p2 = os.path.join(BASE, "final_images", f"{iid}.png")
    print("  iid:", iid, "final_imgs_256:", os.path.exists(p1), "final_images:", os.path.exists(p2))
