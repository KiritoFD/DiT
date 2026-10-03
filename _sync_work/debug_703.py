"""排查 '703:4' (苏轼/隶) 为什么不在词表。"""
import csv
import json
import os
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")

# 1) 训练集里「苏轼-隶」有多少样本
rows = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                                encoding="utf-8")))
su = [r for r in rows if r["calligrapher"] == "苏轼"]
print(f"  训练集 苏轼: {len(su)} 条")
print(f"    按书体: {dict(Counter(r['script'] for r in su))}")
print(f"    苏轼 的 calligrapher_id: {set(r['calligrapher_id'] for r in su)}")
print(f"    (苏轼,隶) 样本数: {len([r for r in su if r['script'] == '隶'])}")

# 2) 词表
m = json.load(open("assets/callig_script_id_map.json", encoding="utf-8"))
pm = m["pair_map"]
print(f"\n  词表 pair_map: {len(pm)} 条, min_samples={m.get('min_samples')}")
# 找 苏轼 的 callig id
su_ids = set(r["calligrapher_id"] for r in su)
print(f"  苏轼 在词表里的 pair: "
      f"{[k for k in pm if k.split(':')[0] in su_ids]}")
print(f"  '703:4' 在词表: {'703:4' in pm}")
print(f"  pair_to_callig 类型: {type(m.get('pair_to_callig')).__name__}")
sparse = m.get("sparse_pairs")
print(f"  sparse_pairs: {sparse if not isinstance(sparse, list) else len(sparse)}")
if isinstance(sparse, list) and sparse:
    print(f"    前5: {sparse[:5]}")

# 3) eval 集里「苏轼-隶」
for f in ("assets/eval_v13_strict_fixed.csv", "assets/eval_v13_seen_fixed.csv"):
    if not os.path.exists(f):
        continue
    ev = list(csv.DictReader(open(f, encoding="utf-8")))
    bad = [r for r in ev
           if f"{r['calligrapher_id']}:{r['script_id']}" not in pm]
    print(f"\n  {os.path.basename(f)}: {len(bad)}/{len(ev)} 条 pair 不在词表")
    for r in bad[:8]:
        print(f"    {r['calligrapher']}/{r['script']}/{r['character']} "
              f"(id {r['calligrapher_id']}:{r['script_id']})")

# 4) eval 集是怎么生成的？看它的 source 列
ev = list(csv.DictReader(open("assets/eval_v13_strict_fixed.csv",
                              encoding="utf-8")))
print(f"\n  eval strict 的 source 分布: "
      f"{dict(Counter(r.get('source', '?') for r in ev))}")
print(f"  样例: {[(r['calligrapher'], r['script'], r['source']) for r in ev[:5]]}")
