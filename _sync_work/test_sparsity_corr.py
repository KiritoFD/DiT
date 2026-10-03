"""决定性检验：strict 的 ssim 是否与「该字在训练集里被多少书家写过」正相关？

若正相关 -> 主因是**数据稀疏**（见过的写法越多，泛化越好）
若无关   -> 主因是别的（容量 / 条件机制 / 采样）
"""
import csv
import os
import statistics as S
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")

BATCH = "/tmp/_eval155only/20260918-081332-v13-base-50k/eval_only/eval_stdskel_batch.csv"
tr = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
st = list(csv.DictReader(open("assets/eval_v13_strict.csv", encoding="utf-8")))
raw = [r for r in csv.DictReader(open(BATCH, encoding="utf-8")) if r["set"] == "strict"]

# 该字被多少个不同书家写过 + 该字总样本数
cals_of_char = defaultdict(set)
n_of_char = defaultdict(int)
for r in tr:
    c = int(r.get("character_id", 0))
    cals_of_char[c].add(int(r["calligrapher_id"]))
    n_of_char[c] += 1

# strict 的第 idx 行 <-> raw 的第 idx 条（同序）
raw_sorted = sorted(raw, key=lambda r: int(r["idx"]))
print(f"  strict {len(st)} 行 / raw {len(raw_sorted)} 条")
pts = []
for i, row in enumerate(st):
    if i >= len(raw_sorted):
        break
    c = int(row.get("character_id", 0))
    pts.append((len(cals_of_char.get(c, set())), n_of_char.get(c, 0),
                float(raw_sorted[i]["ssim"])))

print(f"  可用样本 {len(pts)}")
print()
print("  === 按「该字被多少书家写过」分桶 ===")
buckets = [(0, 1), (2, 3), (4, 7), (8, 15), (16, 100)]
for lo, hi in buckets:
    v = [p[2] for p in pts if lo <= p[0] <= hi]
    if v:
        print(f"    {lo:>3}-{hi:<3} 个书家: n={len(v):>3}  ssim mean={S.mean(v):.4f} "
              f"med={S.median(v):.4f}")

print()
print("  === 按「该字的总样本数」分桶 ===")
for lo, hi in [(1, 5), (6, 20), (21, 60), (61, 200), (201, 99999)]:
    v = [p[2] for p in pts if lo <= p[1] <= hi]
    if v:
        print(f"    {lo:>4}-{hi:<6} 张: n={len(v):>3}  ssim mean={S.mean(v):.4f}")

# Pearson 相关
xs = [p[0] for p in pts]
ys = [p[2] for p in pts]
mx, my = S.mean(xs), S.mean(ys)
cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
r = cov / (sx * sy) if sx * sy else 0
print()
print(f"  Pearson r(书家数, ssim) = {r:+.4f}")

xs2 = [p[1] for p in pts]
mx2 = S.mean(xs2)
cov2 = sum((x - mx2) * (y - my) for x, y in zip(xs2, ys))
sx2 = (sum((x - mx2) ** 2 for x in xs2)) ** 0.5
r2 = cov2 / (sx2 * sy) if sx2 * sy else 0
print(f"  Pearson r(总样本数, ssim) = {r2:+.4f}")
