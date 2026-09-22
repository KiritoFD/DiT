#!/usr/bin/env python
"""分析 VLM 的 GT_CHAR 输出，找出「csv 的 character」与「GT 实际写的字」不一致的样本。

## 判据
  可疑 = s2t(GT_CHAR) != s2t(character)  且 GT_CHAR 不是'？'
简繁归一化用 OpenCC s2t，这样：
  贫/貧、刚/剛、臥/卧  -> 归一化后一致 -> 不算可疑 ✓
  升/陞（异体）        -> 归一化后仍不同 -> 算可疑 ✓（这正是我们要找的）

用法:
  python tools/analyze_vlm_gtchar.py assets/vlm_cmp/strict__4b_v2.csv
"""
import csv
import os
import sys
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")
from opencc import OpenCC  # noqa: E402

_c2t = OpenCC("s2t")


def norm(s):
    return _c2t.convert(s) if s else s


files = sys.argv[1:] or ["assets/vlm_cmp/strict__4b_v2.csv"]
for f in files:
    if not os.path.exists(f):
        print(f"  ✗ {f} 不存在")
        continue
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    n_ok = sum(1 for r in rows if r["gt_char"] not in ("", "？", "?", "None"))
    print(f"\n  === {f} ({len(rows)} 条，VLM 给出字 {n_ok} 条) ===")

    bad = [r for r in rows
           if r["gt_char"] not in ("", "？", "?", "None")
           and norm(r["gt_char"]) != norm(r["char"])]
    eq = [r for r in rows
          if r["gt_char"] not in ("", "？", "?", "None")
          and norm(r["gt_char"]) == norm(r["char"])
          and r["gt_char"] != r["char"]]

    print(f"  ★ 可疑（归一化后仍不同）: {len(bad)}/{len(rows)} "
          f"({len(bad)/max(len(rows),1)*100:.1f}%)")
    print(f"  （另有 {len(eq)} 条只是简繁差异，已排除）")
    print(f"  conf 分布: {dict(Counter(r['conf'] for r in rows).most_common(5))}")
    print(f"\n  {'char':<4}{'vlm':<6}{'conf':<8}{'书家/书体':<16}src")
    for r in bad:
        src = os.path.basename(r.get("src", ""))
        print(f"  {r['char']:<4}{r['gt_char']:<6}{r['conf']:<8}"
              f"{r['callig'] + '/' + r['script']:<16}{src}")
