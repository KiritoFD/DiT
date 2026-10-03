"""诊断简繁错配：csv 的字 / std 字形 / GT 图 三者的简繁一致性。

背景（用户 2026-09-21 发现）:
  有些字明显是繁体，但喂给模型的标准骨架(g)是简体 —— 简繁错配。
  模型被要求"照着简体骨架写出繁体字"，评测时 ssim 也会因简繁不同而偏低。
"""
import csv
import os
import re
from collections import Counter, defaultdict

from opencc import OpenCC

os.chdir("/root/Workspace/xy/DiT")
s2t = OpenCC("s2t")
t2s = OpenCC("t2s")

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
print(f"  csv: {len(rows)} 行")

chars = sorted(set(r["character"] for r in rows))
print(f"  去重字: {len(chars)}")

# 1) 哪些字本身是繁体（t2s 后 != 原字）
trad = [c for c in chars if t2s.convert(c) != c]
simp = [c for c in chars if s2t.convert(c) != c]
both = [c for c in chars if t2s.convert(c) == c and s2t.convert(c) == c]
print(f"\n  === 字符本身的简繁属性 ===")
print(f"    繁体字（t2s 会变）: {len(trad)}")
print(f"    简体字（s2t 会变）: {len(simp)}")
print(f"    简繁同形: {len(both)}")
print(f"    繁体样例: {trad[:20]}")
print(f"    简体样例: {simp[:20]}")

# 2) 关键：csv 里 (script, char) 与 std_path 的对应
#    std_path 指向 data/50k/std/{gid}.png —— 那是"标准字形"
#    如果 char 是繁体、而 std 图是简体 -> 错配
#    检测方法: 看同一个 gid 是否被多个不同的 char 共用（简繁两版共用一个 gid = 错配）
gid2chars = defaultdict(set)
for r in rows:
    m = re.search(r"(\d+)", os.path.basename(r.get("std_path", "")))
    if m:
        gid2chars[int(m.group(1))].add(r["character"])

share = {g: cs for g, cs in gid2chars.items() if len(cs) > 1}
print(f"\n  === std_path(gid) 被多个字共用 ===")
print(f"    gid 总数: {len(gid2chars)}, 多字共用: {len(share)}")
for g, cs in list(share.items())[:15]:
    cs = sorted(cs)
    is_s2t = any(t2s.convert(c) != c for c in cs)
    mark = " ← 疑似简繁共用" if is_s2t else ""
    print(f"    gid {g}: {cs}{mark}")

# 3) 繁体字在训练集里的占比
n_trad_rows = sum(1 for r in rows if t2s.convert(r["character"]) != r["character"])
print(f"\n  === 训练集里繁体字占比 ===")
print(f"    繁体样本: {n_trad_rows}/{len(rows)} = {n_trad_rows/len(rows)*100:.1f}%")
c = Counter(r["script"] for r in rows if t2s.convert(r["character"]) != r["character"])
print(f"    按书体: {dict(c)}")

# 4) 抽样：繁体字 vs 它的简体对应
print(f"\n  === 繁体字及其简体对应（前15）===")
for t in trad[:15]:
    s = t2s.convert(t)
    # 简体版是否也在字表里
    in_simp = s in chars
    print(f"    {t} -> {s}   (简体版在字表: {'是' if in_simp else '否'})")
