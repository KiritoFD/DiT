#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同一个字的 std 骨架，是否随书体不同而不同？决定 few-shot 借 std 时要不要匹配书体。"""
import collections, csv, os
os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
c2s = collections.defaultdict(set)
c2script = collections.defaultdict(set)
for r in rows:
    c2s[r["character"]].add(os.path.basename(r["std_path"]))
    c2script[r["character"]].add(r["script_id"])
multi = {c: v for c, v in c2s.items() if len(v) > 1}
print(f"字数={len(c2s)}  一个字多个 std 的字数={len(multi)}")
for c, v in list(multi.items())[:5]:
    print("   ", c, sorted(v)[:6], "scripts=", sorted(c2script[c]))
one = {c: next(iter(v)) for c, v in c2s.items() if len(v) == 1}
print("单 std 示例:", list(one.items())[:5])
# glyph_id 语义核对
g = collections.defaultdict(set)
for r in rows[:20000]:
    g[r["character"]].add((r["glyph_id"], r["character_id"], r["script_id"]))
print("同字不同 glyph_id 的数量:", sum(1 for c, v in g.items() if len(v) > 1), "/", len(g))
