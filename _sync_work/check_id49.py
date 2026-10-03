"""查 raw id 49（以及其它'只在旧/base、不在新 50k'的书家）到底有多少样本。

判断标准：
  样本数很少 -> 可能是数据质量过滤掉的（合理）
  样本数不少 -> 可能是漏了（需要修）
"""
import csv
import os
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")

CAND = [49, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010]

old = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
new = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))

old_cnt = Counter(int(r["calligrapher_id"]) for r in old)
new_cnt = Counter(int(r["calligrapher_id"]) for r in new)

name_old = {int(r["calligrapher_id"]): r["calligrapher"] for r in old}
name_new = {int(r["calligrapher_id"]): r["calligrapher"] for r in new}

print(f"  {'raw id':>7} {'旧样本':>8} {'新样本':>8}  名字")
for c in CAND:
    print(f"  {c:>7} {old_cnt.get(c, 0):>8} {new_cnt.get(c, 0):>8}  "
          f"{name_old.get(c) or name_new.get(c) or '?'}")

print()
print("=== 旧数据集里样本数最少的几个书家 ===")
for c, n in sorted(old_cnt.items(), key=lambda x: x[1])[:8]:
    print(f"  raw={c:>5}  {n:>5} 样本  {name_old.get(c, '?')}"
          f"  -> 在 50k 里: {'有' if c in new_cnt else '**没有**'}")

print()
print("=== 新 50k 里样本数最少的几个书家 ===")
for c, n in sorted(new_cnt.items(), key=lambda x: x[1])[:6]:
    print(f"  raw={c:>5}  {n:>5} 样本  {name_new.get(c, '?')}")
