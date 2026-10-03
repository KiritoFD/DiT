# -*- coding: utf-8 -*-
"""_scan_polarity_full.py — 全库底色体检: 有多少张是"黑底"(需反相或删除).

判据 (比旧的 border/center 判据鲁棒):
  背景色 = 灰度直方图**众数** mode (书法图背景像素占绝大多数)。
  mode < 128  -> 黑底 (反色图 / 拓片)  ✗
  反相后前景 fg = (a>128).mean()
     0.02 <= fg <= 0.60  -> 有字形, **反相可救** ✓
     其它                -> 无内容(死黑) 或 反相后成墨块 -> **须删除** ✗
输出: /tmp/polarity_bad.csv + 终端统计
"""
import csv
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV = sys.argv[1] if len(sys.argv) > 1 else "assets/train_base_noaug.csv"


def probe(r):
    p = r.get("image_path", "")
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception as e:
        return (p, r.get("character", ""), r.get("calligrapher", ""), r.get("script", ""),
                -1, 0.0, 0.0, "read_error")
    h = np.bincount(a.ravel(), minlength=256)
    mode = int(h.argmax())
    mode_ratio = float(h[mode]) / a.size
    bright = float((a > 128).mean())     # 反相后前景占比
    if mode >= 128:
        v = "ok"
    elif 0.02 <= bright <= 0.60:
        v = "dark_rescuable"
    else:
        v = "dark_hopeless"
    return (p, r.get("character", ""), r.get("calligrapher", ""), r.get("script", ""),
            mode, mode_ratio, bright, v)


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"[scan] {CSV}: {len(rows)} rows", flush=True)
    t0 = time.time()
    res = []
    with mp.Pool(40) as pool:
        for n, rr in enumerate(pool.imap_unordered(probe, rows, chunksize=256), 1):
            res.append(rr)
            if n % 10000 == 0:
                print(f"  {n}/{len(rows)} ({n/(time.time()-t0):.0f}/s)", flush=True)

    cnt = Counter(r[7] for r in res)
    print(f"\n=== 结果 ({time.time()-t0:.0f}s) ===")
    tot = len(res)
    for k in ("ok", "dark_rescuable", "dark_hopeless", "read_error"):
        c = cnt.get(k, 0)
        print(f"  {k:18s} {c:7d}  ({100*c/tot:6.2f}%)")
    dark = cnt.get("dark_rescuable", 0) + cnt.get("dark_hopeless", 0)
    print(f"  {'黑底合计':18s} {dark:7d}  ({100*dark/tot:6.2f}%)")

    # 按数据源
    def src_of(p):
        parts = p.split("/")
        return parts[2] if len(parts) > 2 else "?"
    by = {}
    for r in res:
        s = src_of(r[0])
        d = by.setdefault(s, {"tot": 0, "dark_rescuable": 0, "dark_hopeless": 0})
        d["tot"] += 1
        if r[7] in ("dark_rescuable", "dark_hopeless"):
            d[r[7]] += 1
    print("\n  按数据源:")
    for s, d in sorted(by.items(), key=lambda x: -x[1]["tot"]):
        dr, dh = d["dark_rescuable"], d["dark_hopeless"]
        print(f"    {s:22s} 总={d['tot']:6d} 反色可救={dr:5d} 须删={dh:5d} "
              f"(黑底占比 {100*(dr+dh)/d['tot']:5.2f}%)")

    # 按书家 (坏图集中的)
    byc = Counter()
    for r in res:
        if r[7].startswith("dark"):
            byc[r[2]] += 1
    print("\n  黑底按书家 Top15:")
    for c, n in byc.most_common(15):
        print(f"    {c or '(空)':14s} {n:5d}")

    # 死黑样例
    hop = [r for r in res if r[7] == "dark_hopeless"]
    print(f"\n  死黑样例 (mode, 反相后前景):")
    for r in hop[:10]:
        print(f"    {r[0]}  mode={r[4]:3d} ratio={r[5]:.2f} bright={r[6]:.4f} "
              f"char={r[1]} {r[2]}")

    with open("/tmp/polarity_bad.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "character", "calligrapher", "script",
                    "mode", "mode_ratio", "bright_after_invert", "verdict"])
        for r in res:
            if r[7] != "ok":
                w.writerow(r)
    print(f"\n  -> /tmp/polarity_bad.csv ({dark} 行)")


if __name__ == "__main__":
    main()
