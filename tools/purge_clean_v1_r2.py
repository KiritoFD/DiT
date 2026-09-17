# -*- coding: utf-8 -*-
"""purge_clean_v1_r2.py — 第二轮净化: 用"图 vs 标准字 ink 比值"判据.

动机: 前两轮(55px 宽度 / blob>0.30 / ink<0.01)仍漏两类——
  * 灰块/淡污渍图 (ink≈0.013, 刚好在 0.01 之上, 且 blob 不大)
  * 部分黑块 (blob 未达 0.30)
单看"图"无法区分"淡污渍"和"淡笔画", 但**标准字是干净渲染的**, 可作参照:
  两者构图归一化一致 (实测图中位 ink 0.1815 vs 标准字 0.1828), 故
     ratio = ink_img / ink_std   应接近 1
     ratio 过小 -> 图缺笔画/淡污渍;  ratio 过大 -> 图有墨块/死黑

判据 (任一命中即剔):
  F1 ink_img < 0.030                    过淡
  F2 blob_img > 0.20                    大暗块
  F3 ratio < 0.45 or ratio > 2.20       与标准字不匹配
用法: python tools/purge_clean_v1_r2.py [--apply]
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

SRC = "assets/train_clean_v1_final.csv"
OUT = "assets/train_clean_v1_final.csv"      # 原地收紧
QUAR = "data/_quarantine_v4"
INK_LO, BLOB_HI = 0.030, 0.20
RATIO_LO, RATIO_HI = 0.45, 2.20


def _ink_blob(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (-1.0, -1.0)
    m = a < 128
    ink = float(m.mean())
    blob = 0.0
    if m.any():
        lb, _ = label(m)
        sz = np.bincount(lb.ravel())[1:]
        blob = float(sz.max()) / a.size
    return (ink, blob)


def probe(task):
    k, ip, sp = task
    ii, ib = _ink_blob(ip)
    si, _ = _ink_blob(sp)
    if ii < 0 or si < 0:
        return (k, ii, si, ib, 0.0, "read_error")
    r = ii / max(si, 1e-6)
    rs = []
    if ii < INK_LO:
        rs.append("F1_faint")
    if ib > BLOB_HI:
        rs.append("F2_blob")
    if r < RATIO_LO or r > RATIO_HI:
        rs.append("F3_ratio")
    return (k, ii, si, ib, r, "+".join(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    print(f"[purge2] {SRC}: {len(rows)} 行  apply={a.apply}")
    tasks = [(k, r["image_path"], r["std_path"]) for k, r in enumerate(rows)]
    with mp.Pool(40) as pool:
        res = pool.map(probe, tasks, chunksize=128)

    good = [r[0] for r in res if not r[5]]
    bad = [r[0] for r in res if r[5]]
    c = Counter(r[5] for r in res if r[5])
    print(f"\n  clean {len(good)}  bad {len(bad)}")
    for k, v in c.most_common():
        print(f"    {k:16s} {v}")
    print(f"\n  ★ 最终 = {len(good)}")

    ratios = np.array([r[4] for r in res if r[4] > 0])
    print(f"  ratio: p1={np.percentile(ratios,1):.3f} "
          f"p50={np.percentile(ratios,50):.3f} p99={np.percentile(ratios,99):.3f}")

    if not a.apply:
        print("[DRY-RUN] 加 --apply 执行。")
        return

    os.makedirs(QUAR, exist_ok=True)
    moved = 0
    for k in bad:
        p = rows[k]["image_path"]
        if os.path.exists(p):
            try:
                shutil.move(p, os.path.join(QUAR, os.path.basename(p)))
                moved += 1
            except Exception:
                pass
    print(f"[apply] moved {moved} -> {QUAR}")

    fields = list(rows[0].keys())
    with open(OUT + ".tmp", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k in good:
            w.writerow(rows[k])
    os.replace(OUT + ".tmp", OUT)
    n = sum(1 for _ in open(OUT, encoding="utf-8")) - 1
    print(f"[apply] {OUT}: {n} 行")


if __name__ == "__main__":
    main()
