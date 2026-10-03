# -*- coding: utf-8 -*-
"""确认赵孟/赵孟頫是否同一人，柳公权/柳公權合并影响；输出书体与字量分布"""
import csv
from collections import Counter

TARGETS = ("赵孟", "赵孟頫", "柳公权", "柳公權")

cnt = Counter()
script = Counter()
chars = {}
ids = {}

with open("train.csv", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        c = row["calligrapher"]
        if c in TARGETS:
            cnt[c] += 1
            script[(c, row["script"])] += 1
            chars.setdefault(c, set()).add(row["character"])
            ids[c] = row["calligrapher_id"]

print("=== 样本量 ===")
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"{k}  id={ids[k]}  样本={v}  独字={len(chars[k])}")

print("\n=== 书体分布 ===")
for (c, s), v in sorted(script.items(), key=lambda x: (-x[1], x[0][0])):
    print(f"{c} | {s} : {v}")

print("\n=== 合并对比 ===")
zm = cnt["赵孟"] + cnt["赵孟頫"]
print(f"赵孟+赵孟頫 合并样本 = {zm}（原书家第一名王羲之 {cnt['王羲之'] if '王羲之' in cnt else '?'}）")
