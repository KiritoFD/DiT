"""查 50k 数据集的书家构成：它相对旧数据集到底是多了还是少了。"""
import csv
import json
import os
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")


def load(path, name):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ids = set(int(r["calligrapher_id"]) for r in rows)
    print(f"  {name:<28} {len(rows):>6} 行, {len(ids):>3} 书家")
    return rows, ids


print("=== 三个数据集的规模 ===")
r50, s50 = load("assets/train_50k.csv", "新 50k")
rold, sold = load("assets/train_fame-kxl-tj-px60.csv", "旧 fame-kxl-tj-px60")
try:
    rbase, sbase = load("assets/train_base_sym.csv", "base_sym")
except FileNotFoundError:
    rbase, sbase = [], set()
    print("  (无 train_base_sym.csv)")

print()
print("=== 50k 的 source 列（它是怎么拼出来的）===")
for src, n in Counter(r.get("source", "?") for r in r50).most_common():
    ids = set(int(r["calligrapher_id"]) for r in r50 if r.get("source") == src)
    print(f"  {src:<28} {n:>6} 行, {len(ids):>3} 书家")

print()
print("=== 集合关系 ===")
print(f"  旧 ⊂ 新 ? {sold <= s50}   (旧独有 {len(sold - s50)} 个: {sorted(sold - s50)[:10]})")
print(f"  新 ⊂ 旧 ? {s50 <= sold}   (新独有 {len(s50 - sold)} 个: {sorted(s50 - sold)[:10]})")
if sbase:
    print(f"  base ⊂ 新 ? {sbase <= s50}  (base独有 {len(sbase - s50)} 个: {sorted(sbase - s50)[:12]})")
    print(f"  旧 ⊂ base ? {sold <= sbase}  (旧独有 {len(sold - sbase)} 个)")

print()
print("=== 评测集用到的书家在哪 ===")
ev = set()
for p in ("assets/eval_seen_v10.csv", "assets/eval_fame3_strict_clean_v9.csv"):
    ev |= set(int(r["calligrapher_id"]) for r in csv.DictReader(open(p, encoding="utf-8")))
print(f"  评测集书家: {len(ev)} 个")
print(f"    在新 50k 里: {len(ev & s50)}")
print(f"    只在 base 里: {len(ev & sbase - s50)}")
print(f"    两边都没有: {len(ev - s50 - sbase)} -> {sorted(ev - s50 - sbase)[:10]}")
