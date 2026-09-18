"""决定性检验：书家间的 strict 差异，是由【数据量】还是【风格条件强度】解释的？

对每个书家 c：
  A) 训练样本数 n_train(c)
  B) 风格条件强度 style(c)   （diversity_inter_by_callig，越大=换书家输出变得越多）
  C) strict ssim(c)

算 C 与 A、C 与 B 的相关。谁相关谁就是瓶颈。
"""
import csv
import os
from collections import Counter, defaultdict

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

# C: strict 逐书家
sr = list(csv.DictReader(open("assets/eval_v13_strict.csv", encoding="utf-8")))
batch = [r for r in csv.DictReader(open(
    "assets/results/v13_base_50k/eval_stdskel_batch.csv", encoding="utf-8"))
    if r["set"] == "strict"]
st = max(int(r["step"]) for r in batch)
batch = sorted([r for r in batch if int(r["step"]) == st], key=lambda r: int(r["idx"]))
by = defaultdict(list)
for r in batch:
    i = int(r["idx"])
    if i < len(sr):
        by[sr[i]["calligrapher"]].append(float(r["ssim"]))
C = {c: float(np.mean(v)) for c, v in by.items()}
print(f"  C: strict ssim，{len(C)} 个书家 (step={st})")

# A: 训练样本数
tr = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
A = dict(Counter(r["calligrapher"] for r in tr))
print(f"  A: 训练样本数，{len(A)} 个书家")

# A2: 每书家覆盖的字符数（另一种"数据量"）
A2 = {}
_cs = defaultdict(set)
for r in tr:
    _cs[r["calligrapher"]].add(r["character"])
A2 = {c: len(v) for c, v in _cs.items()}
print(f"  A2: 每书家覆盖字符数")

# B: 风格强度（旧 pairs 文件，第二列才是书家）
pairs = list(csv.DictReader(open(
    "assets/diversity_inter_pairs_v13_155k_percalig.csv", encoding="utf-8")))
new_fmt = "char_a" in set(pairs[0].keys())
ka, kb = ("callig_a", "callig_b") if new_fmt else ("glyph_a", "glyph_b")
B = defaultdict(list)
for r in pairs:
    if r["kind"] == "callig":
        d = 1.0 - float(r["ssim"])
        B[r[ka]].append(d)
        B[r[kb]].append(d)
B = {c: float(np.mean(v)) for c, v in B.items()}
print(f"  B: 风格强度，{len(B)} 个书家")


def corr(xs, ys):
    x, y = np.array(xs), np.array(ys)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    k, b = np.polyfit(x, y, 1)
    return r, k


print()
print("  === 书家间 strict 差异由谁解释 ===")
for name, D in (("A  训练样本数", A), ("A2 覆盖字符数", A2), ("B  风格强度", B)):
    common = sorted(set(C) & set(D))
    if len(common) < 3:
        print(f"    {name}: 交集只有 {len(common)}，跳过")
        continue
    r, k = corr([D[c] for c in common], [C[c] for c in common])
    print(f"    {name:<14} n={len(common):>3}  r={r:+.4f}  斜率={k:+.2e}")

common = sorted(set(C) & set(A) & set(B))
print()
print(f"  === 三边都有: {len(common)} 个书家 ===")
print(f"  {'书家':<10}{'n_train':>8}{'覆盖字':>7}{'风格':>8}{'strict':>8}")
for c in sorted(common, key=lambda z: -C[z]):
    print(f"  {c:<10}{A[c]:>8}{A2[c]:>7}{B[c]:>8.4f}{C[c]:>8.4f}")
