#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys

p = sys.argv[1]
c = json.load(open(p, encoding="utf-8"))
print(f"=== {p} ===")
for k in sorted(c.keys()):
    if k.startswith("_"):
        continue
    v = c[k]
    if isinstance(v, (int, float, bool, str)) or v is None:
        print(f"{k} = {v}")
print("\n=== 非标量键 ===")
for k in sorted(c.keys()):
    if k.startswith("_"):
        continue
    if not isinstance(c[k], (int, float, bool, str)) and c[k] is not None:
        print(f"{k}: {type(c[k]).__name__} = {str(c[k])[:200]}")
