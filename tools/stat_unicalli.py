# -*- coding: utf-8 -*-
"""stat UniCalli data.csv: chars, styles, calligraphers."""
import ast
import collections
import csv
import json

rows = list(csv.DictReader(open("data/unicalli/data.csv", encoding="utf-8")))
out = open("unicalli_stats.txt", "w", encoding="utf-8")
n_chars = 0
styles = collections.Counter()
calligs = collections.Counter()
sc_pairs = collections.Counter()
bad = 0
for r in rows:
    try:
        boxes = ast.literal_eval(r["location"])
    except Exception:
        bad += 1
        continue
    n = len(boxes)
    n_chars += n
    styles[r["chirography"]] += n
    calligs[r["author"]] += n
    sc_pairs[(r["author"], r["chirography"])] += n
out.write(f"rows(作品列): {len(rows)-bad}, bad: {bad}\n")
out.write(f"total chars (bbox): {n_chars}\n\n")
out.write("styles: " + ", ".join(f"{k}:{v}" for k, v in styles.most_common()) + "\n\n")
out.write(f"calligraphers ({len(calligs)}):\n")
for c, v in calligs.most_common():
    out.write(f"  {c}: {v}\n")
out.close()
print("done")
