# -*- coding: utf-8 -*-
"""purge_clean_v1.py — 剔除 clean_v1 里的"空白图"与"整块黑图", 生成最终 csv.

判据 (补齐前几轮的漏检):
  E1 blank : ink < 0.01                 -> 无笔画
  E2 blob  : 最大暗连通域占比 > 0.30     -> 整块黑
两者都**不受 55px 判据覆盖**, 是历史漏网的主因。
动作: 坏图 -> data/_quarantine_v3/ ; 输出 assets/train_clean_v1_final.csv
用法: python tools/purge_clean_v1.py [--apply]
"""
import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "assets/train_clean_v1.csv"
OUT = "assets/train_clean_v1_final.csv"
QUAR = "data/_quarantine_v3"
INK_LO, BLOB_HI = 0.01, 0.30


def probe(task):
    k, p = task
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (k, -1.0, -1.0, "read_error")
    m = a < 128
    ink = float(m.mean())
    blob = 0.0
    if m.any():
        lb, _ = label(m)
        sz = np.bincount(lb.ravel())[1:]
        blob = float(sz.max()) / a.size
    rs = []
    if ink < INK_LO:
        rs.append("E1_blank")
    if blob > BLOB_HI:
        rs.append("E2_blob")
    return (k, ink, blob, "+".join(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    print(f"[purge] {SRC}: {len(rows)} 行  apply={a.apply}")
    tasks = list(enumerate(r["image_path"] for r in rows))
    with mp.Pool(40) as pool:
        res = pool.map(probe, tasks, chunksize=128)

    bad_idx = [r[0] for r in res if r[3]]
    good_idx = [r[0] for r in res if not r[3]]
    c = Counter(r[3] for r in res if r[3])
    print(f"\n  clean {len(good_idx)}  bad {len(bad_idx)}")
    for k, v in c.most_common():
        print(f"    {k:24s} {v}")
    print(f"\n  ★ 最终可用 = {len(good_idx)}")

    if not a.apply:
        print("[DRY-RUN] 加 --apply 执行。")
        return

    os.makedirs(QUAR, exist_ok=True)
    moved = 0
    for k in bad_idx:
        p = rows[k]["image_path"]
        if os.path.exists(p):
            dst = os.path.join(QUAR, os.path.basename(p))
            try:
                shutil.move(p, dst)
                moved += 1
            except Exception:
                pass
    print(f"[apply] moved {moved} -> {QUAR}")

    fields = list(rows[0].keys())
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k in good_idx:
            w.writerow(rows[k])
    n = sum(1 for _ in open(OUT, encoding="utf-8")) - 1
    print(f"[apply] {OUT}: {n} 行")


if __name__ == "__main__":
    main()
