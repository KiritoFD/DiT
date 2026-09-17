# -*- coding: utf-8 -*-
"""stat_tongji.py — 查清 tongji 到底有多少可用样本。

疑问: data/imgs/calli_tongji_imgs/ 有 2,992 张, 但 train_base_noaug.csv 只收了
      2,092 行, 差 900。另有 assets/train_tongji_only.csv 和 data/imgs/calli_tongji/。
"""
import csv
import glob
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ids_of(paths):
    out = []
    for p in paths:
        b = os.path.basename(p)[:-4]
        if b.isdigit():
            out.append(int(b))
    return sorted(out)


noaug = [r["image_path"] for r in csv.DictReader(
    open("assets/train_base_noaug.csv", encoding="utf-8"))
    if "tongji" in r["image_path"]]
i1 = ids_of(noaug)
print(f"train_base_noaug.csv 中 tongji : {len(noaug)} 行, "
      f"id [{i1[0]}..{i1[-1]}] 唯一 {len(set(i1))}")

if os.path.exists("assets/train_tongji_only.csv"):
    r2 = list(csv.DictReader(open("assets/train_tongji_only.csv", encoding="utf-8")))
    i2 = ids_of([r["image_path"] for r in r2])
    print(f"train_tongji_only.csv        : {len(r2)} 行, "
          f"id [{i2[0]}..{i2[-1]}] 唯一 {len(set(i2))}")
else:
    i2 = []

for d in ("data/imgs/calli_tongji", "data/imgs/calli_tongji_imgs",
          "data/imgs/calli_tongji_sym"):
    if os.path.isdir(d):
        fs = glob.glob(os.path.join(d, "*.png"))
        ii = ids_of(fs)
        print(f"{d:32s}: {len(fs)} 张" + (f", id [{ii[0]}..{ii[-1]}]" if ii else ""))

s1, s2 = set(i1), set(i2)
if s2:
    print(f"\n  tongji_only 有而 noaug 没有 : {len(s2 - s1)}")
    print(f"  noaug 有而 tongji_only 没有 : {len(s1 - s2)}")
    miss = sorted(s2 - s1)[:10]
    print(f"  差集样例: {miss}")
    # 这些 id 的图还在吗?
    exist = sum(1 for i in s2 - s1 if os.path.exists(f"data/imgs/calli_tongji_imgs/{i}.png"))
    print(f"  其中图仍存在的: {exist}")

# 书家/书体分布
rows_g = [r for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8"))
          if "tongji" in r["image_path"]]
print(f"\n  noaug tongji 书家: {dict(Counter(r.get('calligrapher','') for r in rows_g).most_common(10))}")
print(f"  noaug tongji 书体: {dict(Counter(r.get('script','') for r in rows_g))}")
if i2:
    r2 = [r for r in csv.DictReader(open("assets/train_tongji_only.csv", encoding="utf-8"))]
    print(f"  tongji_only 书体: {dict(Counter(r.get('script','') for r in r2))}")
