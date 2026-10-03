# -*- coding: utf-8 -*-
"""_clean_scan.py — 找出全部"不干净"的图 (黑底 / 黑框 / 整体偏暗), 只报告不动文件.

判据 (三选一即判脏, 宁多勿少 -> 保证剩余"绝对干净"):
  ink      = (a<128).mean()        > 0.40   # 墨占比过高: 黑底/墨团/死黑
  edge_ink = 边缘 10% 环带的 ink    > 0.50   # 黑 pad 边框 (旧反相修复的副作用)
  med      = median(a)             < 150    # 整体偏暗

输出: /tmp/dirty_base.csv, /tmp/clean_stats.txt, montage
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

NOAUG = "assets/train_base_noaug.csv"
SYM = "assets/train_base_sym.csv"
OUT = "/root/Workspace/xy/DiT/_otout_clean"
os.makedirs(OUT, exist_ok=True)

INK_HI, EDGE_HI, MED_LO = 0.40, 0.50, 150


def probe(r):
    p = r.get("image_path", "")
    meta = (r.get("character", ""), r.get("calligrapher", ""), r.get("script", ""))
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (p, *meta, -1.0, -1.0, -1.0, "read_error")
    H, W = a.shape
    e = max(1, int(0.10 * min(H, W)))
    edge = np.concatenate([a[:e, :].ravel(), a[-e:, :].ravel(),
                           a[:, :e].ravel(), a[:, -e:].ravel()])
    ink = float((a < 128).mean())
    edge_ink = float((edge < 128).mean())
    med = float(np.median(a))
    reasons = []
    if ink > INK_HI:
        reasons.append("ink")
    if edge_ink > EDGE_HI:
        reasons.append("edge")
    if med < MED_LO:
        reasons.append("dark")
    v = "dirty:" + "+".join(reasons) if reasons else "clean"
    return (p, *meta, ink, edge_ink, med, v)


def montage(paths, out, invert=False):
    CELL, COLS = 96, 8
    n = min(len(paths), 64)
    nrow = max((n + COLS - 1) // COLS, 1)
    cv = Image.new("RGB", (COLS * CELL, nrow * CELL), (128, 0, 0))
    for i, p in enumerate(paths[:n]):
        try:
            a = np.asarray(Image.open(p).convert("L").resize((CELL, CELL)))
            if invert:
                a = 255 - a
            cv.paste(Image.fromarray(a).convert("RGB"),
                     ((i % COLS) * CELL, (i // COLS) * CELL))
        except Exception:
            pass
    cv.save(out)


def main():
    rows = list(csv.DictReader(open(NOAUG, encoding="utf-8")))
    print(f"[scan] {NOAUG}: {len(rows)} 行", flush=True)
    t0 = time.time()
    res = []
    with mp.Pool(40) as pool:
        for n, rr in enumerate(pool.imap_unordered(probe, rows, chunksize=256), 1):
            res.append(rr)
            if n % 20000 == 0:
                print(f"  {n}/{len(rows)} ({n/(time.time()-t0):.0f}/s)", flush=True)

    dirty = [r for r in res if r[-1].startswith("dirty")]
    clean = [r for r in res if r[-1] == "clean"]
    print(f"\n=== base 结果 ({time.time()-t0:.0f}s) ===")
    print(f"  clean {len(clean)} ({100*len(clean)/len(res):.2f}%)")
    print(f"  dirty {len(dirty)} ({100*len(dirty)/len(res):.2f}%)")

    rc = Counter(r[-1] for r in dirty)
    print("\n  脏因分布:")
    for k, v in rc.most_common():
        print(f"    {k:28s} {v}")

    def src_of(p):
        parts = p.split("/")
        return parts[2] if len(parts) > 2 else "?"
    by = {}
    for r in res:
        s = src_of(r[0])
        d = by.setdefault(s, Counter())
        d[r[-1].split(":")[0]] += 1
    print("\n  按数据源:")
    for s, d in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        print(f"    {s:22s} 总={sum(d.values()):6d} clean={d.get('clean',0):6d} "
              f"dirty={d.get('dirty',0):6d} ({100*d.get('dirty',0)/sum(d.values()):5.2f}%)")

    # clean 组的 ink 分布 (确认干净)
    ci = np.array([r[4] for r in clean])
    print(f"\n  clean 组 ink: p50={np.percentile(ci,50):.3f} p99={np.percentile(ci,99):.3f} "
          f"max={ci.max():.3f}")

    montage([r[0] for r in clean], f"{OUT}/clean.png")
    montage([r[0] for r in dirty], f"{OUT}/dirty.png")
    montage([r[0] for r in dirty], f"{OUT}/dirty_inv.png", invert=True)
    print(f"  -> montage {OUT}")

    # 唯一脏图路径 + base 行号
    dirty_paths = {r[0] for r in dirty}
    dirty_idx = set()
    for k, r in enumerate(rows):
        if r["image_path"] in dirty_paths:
            dirty_idx.add(k)
    with open("/tmp/dirty_base.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "character", "calligrapher", "script",
                    "ink", "edge_ink", "median", "verdict"])
        for r in dirty:
            w.writerow(r)
    print(f"  -> /tmp/dirty_base.csv ({len(dirty)} 行)")

    # 预估 sym 影响
    sym = list(csv.DictReader(open(SYM, encoding="utf-8")))
    keep = rm = 0
    for r in sym:
        aug = r.get("aug", "")
        if aug == "":
            keep += 1 if r["image_path"] not in dirty_paths else 0
            rm += 1 if r["image_path"] in dirty_paths else 0
        else:
            uid = int(os.path.basename(r["image_path"])[:-4])
            idx = uid - (7000000 if uid < 7100000 else 7100000)
            if idx in dirty_idx:
                rm += 1
            else:
                keep += 1
    print(f"\n  sym csv: 总={len(sym)} -> 保留≈{keep}, 剔除≈{rm}")
    print(f"\n  ★ 最终干净 base = {len(clean)}   (用户要求 >= 40000)")

    with open("/tmp/clean_stats.txt", "w") as f:
        f.write(f"clean={len(clean)} dirty={len(dirty)} sym_keep={keep} sym_rm={rm}\n")
        for k, v in rc.most_common():
            f.write(f"{k}={v}\n")


if __name__ == "__main__":
    main()
