# -*- coding: utf-8 -*-
"""list tongji new calligraphers (not in fame)."""
import csv
import glob
import os
import zipfile

fame = set()
for p in ["assets/train_fame3_clean_v8.csv", "assets/train_fame3_e_full.csv"]:
    for r in csv.DictReader(open(p, encoding="utf-8")):
        fame.add(r["calligrapher"])
tj = {}
for d in sorted(glob.glob("data/imgs/calli_tongji/*")):
    name = os.path.basename(d)
    c, s = name.rsplit("-", 1)
    tj.setdefault(c, []).append(s)
new = sorted(set(tj) - fame)
with open("tongji_new_calligs.txt", "w", encoding="utf-8") as f:
    for c in new:
        f.write(f"{c}: {tj[c]}\n")
    f.write(f"fame names: {sorted(fame)}\n")
print("done")
