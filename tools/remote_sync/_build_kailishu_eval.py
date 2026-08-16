# -*- coding: utf-8 -*-
"""从楷隶 train 构建一个 hold-out eval 子集 csv(供 auto-eval)。
策略: 按 calligrapher 抽样少量不同字, 且确保有标准字形 latent。
输出: kailishu_eval.csv (~N 行)。
"""
import csv, os, random
from collections import defaultdict

random.seed(1)
SRC = "kailishu_train.csv"
N_PER_CALLIG = 2
MAX_TOTAL = 200

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
by_callig = defaultdict(list)
for r in rows:
    by_callig[r["calligrapher_id"]].append(r)

# 每书家取 N 行, 尽量不同字
out = []
sampled_chars = set()
for cid, rlist in by_callig.items():
    random.shuffle(rlist)
    taken = 0
    for r in rlist:
        if taken >= N_PER_CALLIG:
            break
        key = (r["calligrapher_id"], r["character"])
        if key in sampled_chars:
            continue
        sampled_chars.add(key)
        out.append(r)
        taken += 1
    if len(out) >= MAX_TOTAL:
        break

with open("kailishu_eval.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(out)
print(f"eval 子集: {len(out)} 行 (callig书家 {len(by_callig)}, 覆盖字 {len(sampled_chars)})")
