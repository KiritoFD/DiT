"""核实 character_id vs character(string) 的不一致 —— 这决定了"字数"到底是多少。"""
import csv
import os
from collections import Counter, defaultdict

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))

by_id = defaultdict(set)
by_str = defaultdict(set)
for r in rows:
    by_id[r["character"]].add(int(r["character_id"]))
    by_str[int(r["character_id"])].add(r["character"])

print(f"  唯一 character_id : {len(by_str)}")
print(f"  唯一 character 串 : {len(by_id)}")
multi = {k: v for k, v in by_id.items() if len(v) > 1}
print(f"  一个字符串对应多个 id 的: {len(multi)}")
if multi:
    k = next(iter(multi))
    print(f"    例: {k!r} -> ids {sorted(multi[k])}")
m2 = {k: v for k, v in by_str.items() if len(v) > 1}
print(f"  一个 id 对应多个字符串的: {len(m2)}")
if m2:
    k = next(iter(m2))
    print(f"    例: id {k} -> {sorted(m2[k])}")

print()
print("  === 两种口径下的 (书家,字) 对 ===")
for name, key in (("用 character_id", "character_id"), ("用 character 串", "character")):
    pair = Counter((int(r["calligrapher_id"]), r[key]) for r in rows)
    d1 = sum(1 for v in pair.values() if v == 1)
    print(f"    {name}: 对数 {len(pair)}, 只1张 {d1} ({d1/len(pair)*100:.1f}%)")

print()
print("  === 覆盖率 ===")
for name, key in (("character_id", "character_id"), ("character 串", "character")):
    nc = len(set(r[key] for r in rows))
    ncal = len(set(int(r["calligrapher_id"]) for r in rows))
    pair = Counter((int(r["calligrapher_id"]), r[key]) for r in rows)
    print(f"    {name}: {ncal}书家 x {nc}字 = {ncal*nc} 组合, "
          f"已覆盖 {len(pair)} ({len(pair)/(ncal*nc)*100:.2f}%)")
