# -*- coding: utf-8 -*-
"""purge_by_source.py — 按数据源分层清洗 (关键: fame3/tongji 本就干净, 不可误伤).

分层依据 (来自 assets/train_base_noaug.csv 的 id -> 源):
  fame3 / tongji : 已验证干净的数据集, **只用最保守判据**
                   (H1 空白 ink<0.02, H2 极端黑块 blob>0.60)
  unicalli       : 已知问题集中地, **严格判据**
                   (U1 脏背景 bg<200, U2 大暗块 blob>0.24,
                    U3 过淡 ink<0.05, U4 与标准字比值失配 ratio<0.5 或 >2.0)

这样既清掉 UniCalli 的污渍/黑底, 又不会把 fame3 的粗笔画字当墨块剔掉。
用法: python tools/purge_by_source.py [--apply]
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
OUT = "assets/train_clean_v1_final.csv"
QUAR = "data/_quarantine_v6"

# 分层阈值
CONSERV = dict(ink_lo=0.020, blob_hi=0.60, bg_lo=180.0, r_lo=0.30, r_hi=3.0)
STRICT = dict(ink_lo=0.050, blob_hi=0.24, bg_lo=210.0, r_lo=0.50, r_hi=2.00)


def build_id2src():
    m = {}
    for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
        iid = os.path.basename(r["image_path"])[:-4]
        ns = r["image_path"].split("/")
        m[iid] = ns[2] if len(ns) > 2 else "?"
    return m


def stats(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return None
    m = a < 128
    ink = float(m.mean())
    blob = 0.0
    if m.any():
        lb, _ = label(m)
        sz = np.bincount(lb.ravel())[1:]
        blob = float(sz.max()) / a.size
    b = a[a > 128]
    bg = float(np.percentile(b, 50)) if b.size > 100 else 0.0
    return (ink, blob, bg)


def probe(task):
    k, ip, sp, src = task
    si = stats(ip)
    ss = stats(sp)
    if si is None or ss is None:
        return (k, src, 0, 0, 0, 0, "read_error")
    ink, blob, bg = si
    r = ink / max(ss[0], 1e-6)
    T = STRICT if src == "unicalli_chars" else CONSERV
    rs = []
    if ink < T["ink_lo"]:
        rs.append("faint")
    if blob > T["blob_hi"]:
        rs.append("blob")
    if bg < T["bg_lo"]:
        rs.append("bg")
    if r < T["r_lo"] or r > T["r_hi"]:
        rs.append("ratio")
    return (k, src, ink, blob, bg, r, "+".join(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    id2src = build_id2src()
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    print(f"[purge] {SRC}: {len(rows)} 行")
    tasks = []
    for k, r in enumerate(rows):
        iid = os.path.basename(r["image_path"])[:-4]
        tasks.append((k, r["image_path"], r["std_path"], id2src.get(iid, "?")))

    src_cnt = Counter(t[3] for t in tasks)
    print(f"  分层: {dict(src_cnt)}")

    with mp.Pool(40) as pool:
        res = pool.map(probe, tasks, chunksize=128)

    good = [r[0] for r in res if not r[6]]
    bad = [r[0] for r in res if r[6]]
    print(f"\n  clean {len(good)}  bad {len(bad)}")
    rc = Counter(r[6] for r in res if r[6])
    for k, v in rc.most_common():
        print(f"    {k:28s} {v}")

    print("\n  按源:"    )
    for s in src_cnt:
        tot = sum(1 for r in res if r[1] == s)
        g = sum(1 for r in res if r[1] == s and not r[6])
        print(f"    {s:22s} {tot:6d} -> {g:6d} ({100*g/max(tot,1):5.1f}%)")
    print(f"\n  ★ 最终 = {len(good)}")

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
