# -*- coding: utf-8 -*-
"""统计每书体行数(样本数)+ 唯一callig/char, 评估楷隶子集规模。"""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open("5script/train.csv", encoding="utf-8")))
by = defaultdict(lambda: {"rows":0,"callig":set(),"char":set()})
for r in rows:
    s = r["script"]
    by[s]["rows"] += 1
    by[s]["callig"].add(r["calligrapher_id"])
    by[s]["char"].add(r["character"])

for s, d in sorted(by.items()):
    print(f"{s}: 样本{d['rows']}, 书家{len(d['callig'])}, 字{len(d['char'])}")
# 楷隶合计
kai = by.get("楷", {}); li = by.get("隶", {})
print(f"\n楷+隶: 样本={kai['rows']+li['rows']}, (楷{kai['rows']}+隶{li['rows']})")
