# -*- coding: utf-8 -*-
"""fame(旧) vs fame3 书家集合 diff."""
import collections
import csv

old_calligs = collections.Counter()
old_cs = collections.Counter()
with open("assets/train_fame_clean_v8.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        old_calligs[r["calligrapher"]] += 1
        old_cs[(r["calligrapher"], r["script"])] += 1

f3_calligs = set()
with open("assets/train_fame3_clean_v8.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        f3_calligs.add(r["calligrapher"])

out = open("fame_diff.txt", "w", encoding="utf-8")
out.write(f"old train_fame_clean_v8: {len(old_calligs)} calligraphers, {sum(old_calligs.values())} rows\n")
out.write(f"fame3 clean_v8: {len(f3_calligs)} calligraphers\n\n")
out.write("calligraphers in old but NOT in fame3:\n")
for c, v in old_calligs.most_common():
    if c not in f3_calligs:
        detail = " ".join(f"{s}:{n}" for (cc, s), n in sorted(old_cs.items()) if cc == c)
        out.write(f"  {c} ({v} rows; {detail})\n")
out.close()
print("done")
