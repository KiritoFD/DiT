"""逐条核实那份分析里的数字/文件引用。"""
import csv
import json
import os
from collections import Counter, defaultdict

os.chdir("/root/Workspace/xy/DiT")

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
print(f"训练集 {len(rows)} 行\n")

cals = set(int(r["calligrapher_id"]) for r in rows)
chars = set(r["character"] for r in rows)
scripts = set(r["script"] for r in rows)
print("【② 数据全貌】")
print(f"  不同书家: {len(cals)}   （文中说 45–48）")
print(f"  不同字符: {len(chars)}  （文中说 ~5,943）")
print(f"  不同 script: {len(scripts)}")

# 每字被多少书家写过
cpc = defaultdict(set)
for r in rows:
    cpc[r["character"]].add(int(r["calligrapher_id"]))
n1 = sum(1 for v in cpc.values() if len(v) == 1)
print(f"  只被 1 个书家写过的字: {n1} / {len(chars)} ({n1/len(chars)*100:.1f}%)")

# (书家,字) 对
pair = Counter((int(r["calligrapher_id"]), r["character"]) for r in rows)
d1 = sum(1 for v in pair.values() if v == 1)
print(f"\n【③ (书家,字) 对】")
print(f"  总对数 {len(pair)}")
print(f"  只有 1 张: {d1} ({d1/len(pair)*100:.1f}%)  （文中说 28,235/37,630=75.0%）")
print(f"  平均: {len(rows)/len(pair):.2f} 张/对  （文中说 1.4）")

# 跨书体的书家
cal_scripts = defaultdict(set)
for r in rows:
    cal_scripts[int(r["calligrapher_id"])].add(r["script"])
multi = sum(1 for v in cal_scripts.values() if len(v) > 1)
print(f"\n【④ 跨书体的书家】{multi} / {len(cals)}  （文中说 33/48）")

# 覆盖率
print(f"\n【⑤ 组合覆盖率】")
covered = len(pair)
total_possible = len(cals) * len(chars)
print(f"  已覆盖 {covered} / {total_possible} = {covered/total_possible*100:.2f}%")
print(f"  未监督的组合: {(1-covered/total_possible)*100:.1f}%  （文中说 88%）")

# registry / master_results
print(f"\n【⑥ 结果文件】")
for p in ("assets/master_results.json", "assets/registry/runs.json"):
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        n = len(d) if isinstance(d, (list, dict)) else "?"
        print(f"  {p}: 存在, {n} 条")
    else:
        print(f"  {p}: ✗ 不存在")
