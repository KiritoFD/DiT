"""算书家 id 的并集：训练集 + 评测集 + 旧训练集 + base，看词表该多大。"""
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")

CSVS = {
    "新训练 train_50k": "assets/train_50k.csv",
    "旧训练 fame-kxl-tj-px60": "assets/train_fame-kxl-tj-px60.csv",
    "评测 seen_v10": "assets/eval_seen_v10.csv",
    "评测 strict_clean_v9": "assets/eval_fame3_strict_clean_v9.csv",
}

per = {}
for name, p in CSVS.items():
    if not os.path.exists(p):
        print(f"  (缺) {p}")
        continue
    ids = set(int(r["calligrapher_id"]) for r in csv.DictReader(open(p, encoding="utf-8")))
    per[name] = ids
    print(f"  {name:<26} {len(ids):>4} 个书家")

union = set().union(*per.values())
print()
print(f"  并集 = {len(union)} 个")

base = {int(k) for k in json.load(open("assets/callig_id_map_base.json",
                                      encoding="utf-8"))["id_map"]}
print(f"  base 词表 = {len(base)} 个")
print(f"  并集 vs base: 并集多 {len(union - base)} 个, base 多 {len(base - union)} 个")

new_only = per["新训练 train_50k"] - set().union(*[v for k, v in per.items()
                                                   if k != "新训练 train_50k"])
print(f"  只在**新训练集**里出现的书家: {len(new_only)} 个")

# 评测集里不在新训练集的
ev = per.get("评测 seen_v10", set()) | per.get("评测 strict_clean_v9", set())
only_eval = ev - per["新训练 train_50k"]
print(f"  只在**评测集**里出现的书家: {len(only_eval)} 个 -> {sorted(only_eval)[:12]}")
