"""找出 wild_extract 里**不在 50k 训练集**里的书家 —— few-shot 的候选。"""
import csv
import os
from collections import Counter, defaultdict

os.chdir("/root/Workspace/xy/DiT")
WILD = "/root/Workspace/xy/HCSU/wild_extract"

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
in50 = Counter(r["calligrapher"] for r in rows)
print(f"  50k 里 {len(in50)} 个书家\n")

d = defaultdict(list)
for name in sorted(os.listdir(WILD)):
    p = os.path.join(WILD, name)
    if not os.path.isdir(p):
        continue
    n = len([f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    cal, _, script = name.partition("-")
    d[cal].append((script or "?", n))

print("  === wild 里有、但 50k 里没有的书家 ===")
cands = []
for cal in sorted(d):
    if cal in in50:
        continue
    tot = sum(n for _, n in d[cal])
    scripts = ", ".join(f"{s}:{n}" for s, n in sorted(d[cal], key=lambda x: -x[1]))
    cands.append((cal, tot, scripts))
for cal, tot, s in sorted(cands, key=lambda x: -x[1]):
    print(f"    {cal:<8} 共 {tot:>4} 张   [{s}]")
print(f"\n  候选书家 {len(cands)} 个")

print("\n  === 参考：50k 里样本最少的书家（对比用）===")
for k, v in sorted(in50.items(), key=lambda kv: kv[1])[:6]:
    print(f"    {k:<8} {v}")
