# -*- coding: utf-8 -*-
"""restore_all_quarantine.py — 把所有隔离区的图按原始路径恢复, 回到 54,892 全量.

前几轮清洗(55px 宽度 / blob>0.20 / bg<240)用的是**统一判据**, 把 fame3 里
合法的粗笔画字也当墨块剔了(fame3 27,552 -> 22,679)。这里全部恢复, 改为
**按数据源分层**重新清洗 (见 tools/purge_by_source.py)。
"""
import csv
import glob
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUARS = ["data/_quarantine", "data/_quarantine_v2", "data/_quarantine_v3",
         "data/_quarantine_v4", "data/_quarantine_v5", "data/_quarantine_v6"]

# id -> 原始路径
id2path = {}
for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
    m = re.search(r"(\d+)\.png", r["image_path"])
    if m:
        id2path[m.group(1)] = r["image_path"]

print(f"[restore] id 表 {len(id2path)}")
total = 0
for Q in QUARS:
    if not os.path.isdir(Q):
        continue
    n = miss = 0
    for f in glob.glob(os.path.join(Q, "**", "*.png"), recursive=True):
        iid = os.path.basename(f)[:-4]
        dst = id2path.get(iid)
        if dst is None:
            miss += 1
            continue
        d = os.path.dirname(dst)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(dst):
            shutil.move(f, dst)
            n += 1
    print(f"  {Q}: restored {n}, unknown-id {miss}")
    total += n
print(f"[restore] total {total}")

# 校验
import collections
cnt = collections.Counter()
for p in glob.glob("data/imgs/*/*.png"):
    cnt[p.split("/")[2]] += 1
print(f"\n  data/imgs 各源: {dict(cnt)}")
print(f"  合计 {sum(cnt.values())}")
