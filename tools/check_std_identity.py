# -*- coding: utf-8 -*-
"""check_std_identity.py — 校验同 (script,char) 的所有标准字文件**内容完全一致**.

std 是"每张图一个文件"(与图同 id, 便于 dataloader 按 id 查表), 因此同一
(script,char) 会有多个路径 -> 必须像素一致, 否则说明渲染有抖动/串字。
"""
import csv
import hashlib
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME = "fame-kxl-tj-px60"
CSV = f"assets/train_{NAME}.csv"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
g = defaultdict(set)
for r in rows:
    g[(r["script"], r["character"])].add(r["std_path"])

multi = {k: v for k, v in g.items() if len(v) > 1}
bad = []
h = {}
for k, paths in multi.items():
    hs = set()
    for p in paths:
        if p not in h:
            with open(p, "rb") as f:
                h[p] = hashlib.md5(f.read()).hexdigest()
        hs.add(h[p])
    if len(hs) > 1:
        bad.append((k, len(paths), len(hs)))

print(f"[{NAME}] rows={len(rows)}  (script,char)={len(g)}  std 文件={sum(len(v) for v in g.values())}")
print(f"  同一 (script,char) 有多个 std 文件的: {len(multi)}")
print(f"  ★ 内容不一致的: {len(bad)}  (应为 0)")
for k, n, nh in bad[:15]:
    print(f"    {k} 文件{n} 不同内容{nh}")
