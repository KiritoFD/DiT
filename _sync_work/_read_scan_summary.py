# -*- coding: utf-8 -*-
"""读取扫描 CSV，输出污染现状的深入统计（分级 + 交叉分析）。"""
import os, sys, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows


def col(rows, k):
    return np.array([float(r[k]) for r in rows], dtype=float)


for name, path in [("TRAIN", "assets/scan_train_pollution.csv"),
                   ("EVAL", "assets/scan_eval_pollution.csv")]:
    if not os.path.isfile(path):
        print(f"{name}: missing")
        continue
    rows = load(path)
    N = len(rows)
    print(f"\n{'='*60}")
    print(f"{name}: {N} 张")
    print(f"{'='*60}")

    ink = col(rows, "ink_ratio")
    ncc = col(rows, "n_cc")
    mainf = col(rows, "main_frac")
    smalln = col(rows, "small_cc_n")
    smalla = col(rows, "small_cc_area_ratio")
    midn = col(rows, "mid_cc_n")
    foreign = col(rows, "foreign_area_ratio")
    edge = col(rows, "edge_ink_ratio")
    bar = col(rows, "border_bar")
    blob = col(rows, "edge_blob_area")
    inv = col(rows, "inverted")

    print(f"墨量 ink_ratio:      mean={ink.mean():.4f} median={np.median(ink):.4f} "
          f"p95={np.percentile(ink,95):.4f} max={ink.max():.4f}")
    print(f"连通域数 n_cc:       mean={ncc.mean():.1f} median={np.median(ncc):.0f} "
          f"p95={np.percentile(ncc,95):.0f} max={ncc.max():.0f}")
    print(f"主连通域占比 main_frac: mean={mainf.mean():.4f} median={np.median(mainf):.4f}")

    print(f"\n--- 噪点 (孤立小墨点) ---")
    for thr in (1, 3, 5, 10):
        c = (smalln >= thr).sum()
        print(f"  small_cc_n >= {thr:>2}: {c:>6} 张 ({c/N*100:5.2f}%)")
    for thr in (0.0005, 0.001, 0.002, 0.005):
        c = (smalla >= thr).sum()
        print(f"  small_cc_area_ratio >= {thr:.4f}: {c:>6} 张 ({c/N*100:5.2f}%)")

    print(f"\n--- 脏污 (非主连通域斑块) ---")
    for thr in (0.01, 0.02, 0.05, 0.10):
        c = (foreign >= thr).sum()
        print(f"  foreign_area_ratio >= {thr:.3f}: {c:>6} 张 ({c/N*100:5.2f}%)")
    for thr in (1, 3, 6):
        c = (midn >= thr).sum()
        print(f"  mid_cc_n >= {thr}: {c:>6} 张 ({c/N*100:5.2f}%)")

    print(f"\n--- 连边大片黑 ---")
    for thr in (0.05, 0.10, 0.20, 0.40):
        c = (edge >= thr).sum()
        print(f"  edge_ink_ratio >= {thr:.2f}: {c:>6} 张 ({c/N*100:5.2f}%)")
    c = (bar == 1).sum()
    print(f"  border_bar (实心黑条): {c:>6} 张 ({c/N*100:5.2f}%)")
    c = (blob > 0).sum()
    print(f"  edge_blob_area > 0:    {c:>6} 张 ({c/N*100:5.2f}%)")
    for thr in (0.01, 0.05):
        c = (blob >= thr * 256 * 256).sum()
        print(f"  edge_blob_area >= {thr:.2f}*全图: {c:>6} 张 ({c/N*100:5.2f}%)")

    print(f"\n--- 其他 ---")
    c = (inv == 1).sum()
    print(f"  疑似反相 (ink>0.5): {c:>6} 张 ({c/N*100:5.2f}%)")
    c = (mainf < 0.5).sum()
    print(f"  碎片严重 (main_frac<0.5): {c:>6} 张 ({c/N*100:5.2f}%)")
    c = (ncc >= 20).sum()
    print(f"  连通域极多 (n_cc>=20): {c:>6} 张 ({c/N*100:5.2f}%)")

    # 交叉: 严重污染 (任一指标显著)
    severe = ((smalla >= 0.002) | (foreign >= 0.10) | (edge >= 0.20) |
              (bar == 1) | (inv == 1) | (mainf < 0.5))
    print(f"\n  >>> 需修复(综合判定): {severe.sum()} 张 ({severe.mean()*100:.2f}%)")
    clean = ((smalla < 0.0005) & (foreign < 0.02) & (edge < 0.05) &
             (bar == 0) & (inv == 0) & (mainf > 0.9))
    print(f"  >>> 完全干净:        {clean.sum()} 张 ({clean.mean()*100:.2f}%)")
