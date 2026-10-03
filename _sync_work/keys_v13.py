#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印 v13 系列 config 的**键**(跳过长 _comment), 并和 v25 做差异对比。"""
import json
import os
import sys

os.chdir("/root/Workspace/xy/DiT")


def keys(p):
    c = json.load(open(p, encoding="utf-8"))
    return {k: v for k, v in c.items() if k != "_comment"}


a = keys("src/train/configs/v13_base_50k.json")
b = keys("src/train/configs/v25_stdskel.json")
try:
    c = keys("src/train/configs/v13_12ch_post.json")
except Exception:
    c = {}

print("=== v13_base_50k 的键 ===")
for k in sorted(a):
    print(f"  {k} = {a[k]}")

print("\n=== v13 vs v25 差异 (键: v13 -> v25) ===")
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k, "<无>"), b.get(k, "<无>")
    if va != vb:
        print(f"  {k}: v13={va}  ->  v25={vb}")

print("\n=== v13_12ch_post 与 v13_base 的差异 ===")
for k in sorted(set(a) | set(c)):
    va, vc = a.get(k, "<无>"), c.get(k, "<无>")
    if va != vc:
        print(f"  {k}: base={va}  ->  12ch={vc}")
