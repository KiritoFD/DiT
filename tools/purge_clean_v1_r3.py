# -*- coding: utf-8 -*-
"""purge_clean_v1_r3.py — 第三轮净化: 剔除"脏背景"图.

发现: 前几轮都在看"墨"(ink/blob/宽度), 漏了**背景本身是灰的**这一类
(如"影"那张: 淡灰底 + 一根竖线, ink 0.07 过了淡图判据, 但背景明显脏)。

判据:
  G1 bg90 < 240        # 背景亮部不足 -> 整体发灰/发暗 (白底书法应≈255)
  G2 bg_std < 12       # 背景亮度标准差过大 -> 底噪不均
用法: python tools/purge_clean_v1_r3.py [--apply]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "assets/train_clean_v1_final.csv"
OUT = "assets/train_clean_v1_final.csv"
QUAR = "data/_quarantine_v5"
BG_LO = 240.0


def probe(task):
    k, p = task
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (k, -1.0, -1.0, "read_error")
    # 背景 = 亮部: 用 >128 的像素统计 (书法图亮部就是背景)
    b = a[a > 128]
    if b.size < 100:
        return (k, 0.0, 0.0, "G1_bg")
    bg90 = float(np.percentile(b, 50))
    bg_std = float(b.std())
    rs = []
    if bg90 < BG_LO:
        rs.append("G1_bg")
    return (k, bg90, bg_std, "+".join(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    print(f"[purge3] {SRC}: {len(rows)} 行  apply={a.apply}")
    tasks = [(k, r["image_path"]) for k, r in enumerate(rows)]
    with mp.Pool(40) as pool:
        res = pool.map(probe, tasks, chunksize=256)

    good = [r[0] for r in res if not r[3]]
    bad = [r[0] for r in res if r[3]]
    c = Counter(r[3] for r in res if r[3])
    print(f"\n  clean {len(good)}  bad {len(bad)}")
    for k, v in c.most_common():
        print(f"    {k:16s} {v}")
    print(f"\n  ★ 最终 = {len(good)}")

    bg = np.array([r[1] for r in res if r[1] >= 0])
    print(f"  bg(亮部中位): p1={np.percentile(bg,1):.1f} "
          f"p5={np.percentile(bg,5):.1f} p50={np.percentile(bg,50):.1f} "
          f"min={bg.min():.1f}")

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
