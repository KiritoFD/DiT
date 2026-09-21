"""按书家聚合 strict ssim + 训练样本数 -> 找出"学得好"的书家。

eval 逐样本 CSV 没有 calligrapher 列，用 idx 对齐 assets/eval_v13_strict.csv 取书家。
"""
import csv
import glob
import os
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")

# 1) 训练样本数
tr = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
n_train = defaultdict(int)
n_chars = defaultdict(set)
for r in tr:
    n_train[r["calligrapher"]] += 1
    n_chars[r["calligrapher"]].add(r["character"])
print(f"  训练: {len(tr)} 行, {len(n_train)} 个书家")

# 2) eval 集的书家（按 idx 对齐）
ev = list(csv.DictReader(open("assets/eval_v13_strict.csv", encoding="utf-8")))
print(f"  eval_v13_strict: {len(ev)} 行")

# 3) eval 结果
target = None
for p in glob.glob("assets/results/*/eval_stdskel_batch.csv"):
    if "v13_wd01" in p:
        target = p
        break
if target is None:
    for p in glob.glob("assets/results/*/eval_stdskel_batch.csv"):
        if "v13_base_50k" in p:
            target = p
            break
if target is None:
    cands = glob.glob("assets/results/*/eval_stdskel_batch.csv")
    target = cands[0] if cands else None
print(f"  用 eval 结果: {target}")

rows = list(csv.DictReader(open(target, encoding="utf-8")))
steps = sorted(set(r.get("step", "") for r in rows))
last = steps[-1]
print(f"  最新 step: {last}")

strict_by = defaultdict(list)
for r in rows:
    if r.get("set") != "strict" or r.get("step") != last:
        continue
    try:
        i = int(r["idx"])
        s = float(r.get("ssim", 0))
    except (ValueError, KeyError):
        continue
    if 0 <= i < len(ev):
        c = ev[i]["calligrapher"]
        strict_by[c].append(s)

print(f"\n  === 各书家 (step {last}, strict) ===")
out = []
for c, ss in strict_by.items():
    m = sum(ss) / len(ss)
    out.append((c, m, len(ss), n_train[c], len(n_chars[c])))
out.sort(key=lambda t: -t[1])
print(f"  {'书家':<8}{'strict':>9}{'n_eval':>8}{'n_train':>9}{'覆盖字':>8}")
for c, m, n, nt, nc in out:
    print(f"  {c:<8}{m:>9.4f}{n:>8}{nt:>9}{nc:>8}")

print(f"\n  === 综合推荐（strict 高 且 训练样本多）===")
best = [t for t in out if t[3] >= 1500]
best.sort(key=lambda t: -(t[1] + min(t[3], 3000) / 3000 * 0.05))
for c, m, n, nt, nc in best[:12]:
    print(f"    {c:<8} strict={m:.4f}  训练={nt}  覆盖字={nc}")
