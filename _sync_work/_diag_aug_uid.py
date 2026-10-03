# -*- coding: utf-8 -*-
"""_diag_aug_uid.py — 量化 aug_base_sym.py 的 uid 双重叠加 bug 的影响范围.

bug: tasks 里已传 UID_T+k / UID_N+k, work() 里又 +idx -> 实际 uid = UID_T+2k。
     k>50000 时 tp 的 7000000+2k 溢出到 tn 段(7100000+) -> 覆盖 tn 的图。
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "assets/train_base_noaug.csv"
rows0 = list(csv.DictReader(open(SRC, encoding="utf-8")))
N = len(rows0)
print(f"[src] {SRC}: {N} 行")

# 复现 bug: idx=k
tp = {7000000 + 2 * k: k for k in range(N)}
tn = {7100000 + 2 * k: k for k in range(N)}
inter = set(tp) & set(tn)
print(f"\n[bug 复现] tp uid 数={len(tp)} tn uid 数={len(tn)} 交集(冲突)={len(inter)}")
if inter:
    print(f"  冲突 uid 范围: [{min(inter)}, {max(inter)}]")
    ks = sorted(tp[u] for u in inter)
    print(f"  对应原图行号 k: [{ks[0]}, {ks[-1]}]  (共 {len(ks)} 个)")
    print(f"  等价条件: k > {min(ks)-1}")

# 实测 csv 里的冲突
rows = list(csv.DictReader(open("assets/train_base_sym.csv", encoding="utf-8")))
byid = defaultdict(list)
for r in rows:
    m = re.search(r"(\d+)\.png", r.get("image_path", ""))
    if m:
        byid[int(m.group(1))].append(r)
dup = {k: v for k, v in byid.items() if len(v) > 1}
print(f"\n[csv 实测] 重复 img_id={len(dup)}  涉及行数={sum(len(v) for v in dup.values())}")
augs = Counter()
for k, v in dup.items():
    augs[tuple(sorted(r.get("aug", "") for r in v))] += 1
print(f"  重复行的 aug 组合分布: {augs.most_common(5)}")

# 这些冲突 id 对应的 png 存在吗? 只有一份
exist = sum(1 for k in dup if os.path.exists(f"data/imgs/base_sym/{k}.png"))
print(f"  冲突 id 中有对应 PNG 文件的: {exist}/{len(dup)}")
print(f"\n[结论] 约 {len(dup)} 对 (tp, tn) 共用一个 PNG -> 其中一方的图被覆盖,")
print(f"       这些行(约 {sum(len(v) for v in dup.values())} 行 = "
      f"{100*sum(len(v) for v in dup.values())/len(rows):.2f}%)的 image_path 指向错误内容。")
