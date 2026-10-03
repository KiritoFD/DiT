"""查 base 独有、50k 没有的 10 个书家（49, 9002..9010）在各数据集里的样本数。"""
import csv
import os
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")

CAND = [49, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010]
FILES = {
    "base_sym (161k)": "assets/train_base_sym.csv",
    "旧 fame-kxl-tj-px60": "assets/train_fame-kxl-tj-px60.csv",
    "新 50k": "assets/train_50k.csv",
}

cnts, names = {}, {}
for name, p in FILES.items():
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    cnts[name] = Counter(int(r["calligrapher_id"]) for r in rows)
    for r in rows:
        names.setdefault(int(r["calligrapher_id"]), r["calligrapher"])

print(f"  {'raw':>6} {'名字':<8}", end="")
for k in cnts:
    print(f" {k:>18}", end="")
print()
for c in CAND:
    print(f"  {c:>6} {names.get(c, '?'):<8}", end="")
    for k in cnts:
        print(f" {cnts[k].get(c, 0):>18}", end="")
    print()

print()
print("=== 评测集里那 5 个不在 50k 的书家 ===")
ev_ids = Counter()
for p in ("assets/eval_seen_v10.csv", "assets/eval_fame3_strict_clean_v9.csv"):
    for r in csv.DictReader(open(p, encoding="utf-8")):
        ev_ids[int(r["calligrapher_id"])] += 1
s50 = set(int(r["calligrapher_id"]) for r in
          csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
for c in sorted(set(ev_ids) - s50):
    print(f"  raw={c:>5} {names.get(c, '?'):<8} 评测集里 {ev_ids[c]:>3} 张  |  "
          f"base_sym 里 {cnts.get('base_sym (161k)', Counter()).get(c, 0):>5} 张  |  "
          f"旧数据集里 {cnts.get('旧 fame-kxl-tj-px60', Counter()).get(c, 0):>5} 张")
