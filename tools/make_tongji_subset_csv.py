# -*- coding: utf-8 -*-
"""make_tongji_subset_csv.py - tongji-only rows (for encode pipelines)."""
import csv

n = 0
with open("assets/train_tongji_only.csv", "w", encoding="utf-8", newline="") as f:
    fields = ["image_path", "calligrapher", "script", "character", "calligrapher_id",
              "script_id", "character_id", "glyph_id", "aug"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in csv.DictReader(open("assets/train_fame_tj_kxl.csv", encoding="utf-8")):
        if "calli_tongji" in r["image_path"]:
            w.writerow({k: r[k] for k in fields})
            n += 1
print(f"written assets/train_tongji_only.csv: {n} rows")
