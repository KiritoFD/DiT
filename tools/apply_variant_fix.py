"""把二阶段判定为 strong 的样本，csv 的 character 改成「通行字」。

## 用户意图
书法数据里 GT 写的是**异体字**（刋/吿/徳/歳...），而 csv 标的是**异体字本身**。
改成「通行字」（刊/告/德/岁...）后：
  - csv 的 character  = 通行字
  - std g（从 character 渲染）= 通行字的规整骨架
  - GT（书法）        = 异体字
-> 模型学会「从通行字骨架写出书家的异体写法」—— 这正是书法风格的一部分

## 要同步改的地方（4 处）
1. assets/train_50k_v2_fixed.csv  的 character
2. data/50k/std/{id}.png          重渲染（用通行字）
3. data/50k/shards_std_fixed/     latent 替换
4. （可选）eval csv 同步

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/apply_variant_fix.py \
      --review assets/stage2_review.csv --dry-run 1
"""
import argparse
import csv
import glob
import os

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--review", default="assets/stage2_review.csv")
ap.add_argument("--train", default="assets/train_50k_v2_fixed.csv")
ap.add_argument("--min-conf", type=float, default=0.9)
ap.add_argument("--dry-run", type=int, default=1)
a = ap.parse_args()

# 1) 收集要改的 (idx, old_char, new_char)
rev = list(csv.DictReader(open(a.review, encoding="utf-8")))
todo = []
for r in rev:
    if r["verdict"] != "strong":
        continue
    try:
        conf = float(r["s2_conf"] or 0)
    except ValueError:
        continue
    if conf < a.min_conf:
        continue
    old, new = r["char"], r["s2_pred"]
    if not new or new in ("?", "") or old == new:
        continue
    todo.append((int(r["idx"]), old, new))

print(f"  review: {len(rev)} 条，strong+conf>={a.min_conf}: {len(todo)} 条")
from collections import Counter  # noqa: E402
pairs = Counter((o, n) for _, o, n in todo)
print(f"  不同 (old->new) 组合: {len(pairs)} 种，Top 20:")
for (o, n), c in pairs.most_common(20):
    print(f"    {o} -> {n}   ×{c}")

if a.dry_run:
    print("\n  [dry-run] 未写盘。去掉 --dry-run 0 执行。")
    raise SystemExit(0)

# 2) 改 train csv
rows = list(csv.DictReader(open(a.train, encoding="utf-8")))
cols = list(rows[0].keys())
m = {i: (o, n) for i, o, n in todo}
n_chg = 0
for i, r in enumerate(rows):
    if i in m and r["character"] == m[i][0]:
        r["character"] = m[i][1]
        n_chg += 1
with open(a.train, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"\n  ✓ train csv: 改了 {n_chg} 行")

# 3) 记录改了哪些 id（供后续重渲染 std / 重建 shards）
with open("assets/variant_fixed_ids.csv", "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "old_50k_id", "old_char", "new_char"])
    for i, o, n in todo:
        w.writerow([i, rows[i].get("old_50k_id", ""), o, n])
print(f"  ✓ -> assets/variant_fixed_ids.csv（供重渲染 std / 重建 shards）")
