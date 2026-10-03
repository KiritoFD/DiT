# -*- coding: utf-8 -*-
"""_check_trad.py — 简繁/异体字体检: 数据字符集与标准字(g)的一致性."""
import csv
import importlib.util
import os
import sys
from collections import Counter

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=== 转换库可用性 ===")
for m in ("opencc", "zhconv", "hanziconv"):
    print(f"  {m}: {bool(importlib.util.find_spec(m))}")

rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
print(f"\n=== 数据源分布 (n={len(rows)}) ===")
src = Counter()
for r in rows:
    p = r.get("image_path", "")
    parts = p.split("/")
    src[parts[2] if len(parts) > 2 else p] += 1
for k, v in src.most_common(10):
    print(f"  {k:30s} {v}")

print("\n=== script (书体) 分布 ===")
print("  ", Counter(r["script"] for r in rows).most_common())

chars = [r["character"] for r in rows if r.get("character")]
uniq = sorted(set(chars))
print(f"\n=== 唯一字符 {len(uniq)} ===")

print("\n=== 标准字一致性 ===")
print("  render(ch, script) 用**数据里的 character 原字** + simkai(楷体) 渲染,")
print("  即 g 与 GT 是同一个码位 -> 天生一致 (数据是什么字, 标准字就渲染什么字).")


def simplify(ch):
    """用可用库把 繁体->简体; 无法转换则原样返回."""
    if importlib.util.find_spec("zhconv"):
        import zhconv
        return zhconv.convert(ch, "zh-cn")
    if importlib.util.find_spec("opencc"):
        from opencc import OpenCC
        return OpenCC("t2s").convert(ch)
    return ch


# 只看"t2s 后会变化"的字 = 繁体/异体
trad = []
for ch in uniq:
    s = simplify(ch)
    if s != ch:
        trad.append((ch, s))
print(f"\n=== 繁体/异体候选 (t2s 后变化): {len(trad)} 个 ===")
if trad:
    for ch, s in trad[:40]:
        print(f"  {ch} -> {s}")
else:
    print("  (无库或全部简繁同形)")

# 抽样看具代表性的字符
print("\n=== 字符抽样 ===")
samp = [c for c in uniq if ord(c) > 0x4E00][:60]
print("  " + "".join(samp))

# 统计"同字不同写法"是否被当成独立字 (影响类别数与数据稀释)
if trad:
    m = {s: [] for _, s in trad}
    for ch, s in trad:
        m[s].append(ch)
    dup = {k: v for k, v in m.items() if len(v) > 1}
    print(f"\n=== 同一简体对应多个繁体/异体: {len(dup)} 组 ===")
    for k, v in list(dup.items())[:15]:
        print(f"  {k} <- {v}")
