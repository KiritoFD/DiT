"""exp-std: 从 top10 训练 csv 切出 held-out 200 条。

held-out 名单 == data/top10_style23/eval_real200_cache.pt 的 img_ids（判分缓存同源，
保证"训练没见过的正好是被判分的那 200 条"）。eval csv 的行直接从训练 csv 按 id 取，
列 schema 与训练完全一致。不按任何质量/难度筛选。
"""
import csv
import os

import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
SRC = "exp-std/csv/train_top10_style23_real.csv"
CACHE = "exp-std/data/eval_real200_cache.pt"
TR = "exp-std/csv/train.csv"
EV = "exp-std/csv/eval200.csv"

c = th.load(CACHE, map_location="cpu", weights_only=False)
held = [int(i) for i in c["img_ids"]]
print(f"[cache] held-out n={len(held)} (rows={len(c['rows'])} 一致={len(held) == len(c['rows'])})")

with open(SRC, encoding="utf-8") as f:
    rd = csv.DictReader(f)
    fns = rd.fieldnames
    rows = list(rd)
print(f"[in] {SRC} rows={len(rows)}")

by_id = {int(r["img_id"]): r for r in rows}
missing = [i for i in held if i not in by_id]
print(f"[chk] held-out 中能在训练 csv 找到的 = {len(held) - len(missing)} / {len(held)}"
      + (f" (缺 {missing[:5]})" if missing else ""))
held_set = set(held)
train = [r for r in rows if int(r["img_id"]) not in held_set]
ev = [by_id[i] for i in held if i in by_id]

with open(TR, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fns)
    w.writeheader()
    w.writerows(train)
with open(EV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fns)
    w.writeheader()
    w.writerows(ev)
print(f"[out] {TR} rows={len(train)}")
print(f"[out] {EV} rows={len(ev)}")

cov = {}
for r in ev:
    cov[r["slot_name"]] = cov.get(r["slot_name"], 0) + 1
print(f"[cov] held-out 覆盖 {len(cov)} 个 (书家×书体) 槽位: "
      + ", ".join(f"{k}:{v}" for k, v in sorted(cov.items(), key=lambda x: -x[1])[:12]))
print(f"[cov] 训练集 {len(train)} 条 / 槽位 {len({r['slot_name'] for r in train})}")
