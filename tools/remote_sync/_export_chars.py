# -*- coding: utf-8 -*-
"""导出每 book体 的 unique 字到 UTF-8 文本文件，供本地字体覆盖检测。"""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open("5script/train.csv", encoding="utf-8")))
chars_by_script = defaultdict(set)
for r in rows:
    chars_by_script[int(r["script_id"])].add(r["character"])

name = {0:"kai",1:"zhuan",2:"cao",3:"xing",4:"li"}
for sid, s in chars_by_script.items():
    with open(f"_chars_{name[sid]}.txt", "w", encoding="utf-8") as f:
        f.write("".join(sorted(s)))
    print(f"{name[sid]}: {len(s)} 字 -> _chars_{name[sid]}.txt")
# 全集合
allc = set()
for s in chars_by_script.values():
    allc |= s
with open("_chars_all.txt", "w", encoding="utf-8") as f:
    f.write("".join(sorted(allc)))
print(f"all: {len(allc)} 字 -> _chars_all.txt")
