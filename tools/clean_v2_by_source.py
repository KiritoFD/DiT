# -*- coding: utf-8 -*-
"""clean_v2_by_source.py — 以 train_base_noaug.csv(54,892) 为准, 按源分层清洗.

教训 (2026-09-15): 之前的统一判据(55px 宽度 / blob>0.20)把 fame3 里合法的
粗笔画字当墨块剔了, 导致总量比 fame3 单独还少。正确做法是**分层**:
  fame3 / tongji : 已验证干净 -> 只剔"空白"和"极端黑块"
  unicalli       : 问题集中地 -> 严格剔(脏背景/大暗块/过淡/与标准字比值失配)

输出:
  assets/train_clean_v2.csv        清洗后的 csv (含 source 列)
  /tmp/clean_v2_report.json        统计
用法: python tools/clean_v2_by_source.py [--apply]
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "assets/train_base_noaug.csv"
OUT = "assets/train_clean_v2.csv"

CONSERV = dict(ink_lo=0.015, blob_hi=0.60, bg_lo=170.0)
STRICT = dict(ink_lo=0.050, blob_hi=0.24, bg_lo=205.0)


def probe(task):
    k, p, src = task
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (k, src, 0.0, 0.0, 0.0, "read_error")
    m = a < 128
    ink = float(m.mean())
    blob = 0.0
    if m.any():
        lb, _ = label(m)
        sz = np.bincount(lb.ravel())[1:]
        blob = float(sz.max()) / a.size
    b = a[a > 128]
    bg = float(np.percentile(b, 50)) if b.size > 100 else 0.0
    T = STRICT if src == "unicalli_chars" else CONSERV
    rs = []
    if ink < T["ink_lo"]:
        rs.append("faint")
    if blob > T["blob_hi"]:
        rs.append("blob")
    if bg < T["bg_lo"]:
        rs.append("bg")
    return (k, src, ink, blob, bg, "+".join(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    print(f"[clean_v2] {SRC}: {len(rows)} 行")
    miss = [r for r in rows if not os.path.exists(r["image_path"])]
    print(f"  缺失图 {len(miss)}")
    if miss:
        for r in miss[:5]:
            print(f"    MISS {r['image_path']}")

    tasks = []
    for k, r in enumerate(rows):
        ns = r["image_path"].split("/")
        tasks.append((k, r["image_path"], ns[2] if len(ns) > 2 else "?"))
    print(f"  分层: {dict(Counter(t[2] for t in tasks))}")

    with mp.Pool(40) as pool:
        res = pool.map(probe, tasks, chunksize=256)

    good = [r[0] for r in res if not r[5]]
    bad = [r[0] for r in res if r[5]]
    print(f"\n  clean {len(good)}  bad {len(bad)}")
    for k, v in Counter(r[5] for r in res if r[5]).most_common():
        print(f"    {k:16s} {v}")
    print("\n  按源:")
    for s in sorted({t[2] for t in tasks}):
        tot = sum(1 for r in res if r[1] == s)
        g = sum(1 for r in res if r[1] == s and not r[5])
        print(f"    {s:22s} {tot:6d} -> {g:6d} ({100*g/max(tot,1):5.1f}%)")
    print(f"\n  ★ 最终 = {len(good)}")

    if not a.apply:
        print("[DRY-RUN] 加 --apply 执行。")
        return

    fields = list(rows[0].keys()) + ["source"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k in good:
            r = dict(rows[k])
            ns = r["image_path"].split("/")
            r["source"] = ns[2] if len(ns) > 2 else "?"
            w.writerow(r)
    n = sum(1 for _ in open(OUT, encoding="utf-8")) - 1
    print(f"[apply] {OUT}: {n} 行")


if __name__ == "__main__":
    main()
