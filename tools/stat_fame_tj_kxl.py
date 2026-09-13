# -*- coding: utf-8 -*-
import collections
import csv

c = collections.Counter()
aug = collections.Counter()
n = 0
for r in csv.DictReader(open("assets/train_fame_tj_kxl.csv", encoding="utf-8")):
    c[r["script"]] += 1
    aug[(r["script"], r["aug"])] += 1
    n += 1
with open("fame_tj_kxl_stats.txt", "w", encoding="utf-8") as f:
    f.write(f"total {n}\nscripts: {dict(c)}\n")
    for k, v in sorted(aug.items()):
        f.write(f"  {k}: {v}\n")
print("done")
