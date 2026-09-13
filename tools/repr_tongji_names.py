# -*- coding: utf-8 -*-
"""repr-compare tongji dir names vs fame calligrapher names."""
import csv
import glob
import os

fame = set()
for p in ["assets/train_fame3_clean_v8.csv"]:
    for r in csv.DictReader(open(p, encoding="utf-8")):
        fame.add(r["calligrapher"])
lines = []
for d in sorted(glob.glob("data/imgs/calli_tongji/*")):
    name = os.path.basename(d)
    c, s = name.rsplit("-", 1)
    lines.append(f"{c!r} {s} in_fame={c in fame}")
with open("tongji_repr.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.write("\nfame repr sample: " + repr(sorted(fame)[:5]))
print("done")
