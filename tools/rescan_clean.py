# -*- coding: utf-8 -*-
"""rescan_clean.py — 回滚前两轮清洗, 并按"单笔画粗度"重新扫描给出各阈值数字.

背景: 第 2 轮用了 J2 笔画宽>26px, 在 256x256 下过严 (剔掉 70%), 用户裁定
      应为 **单笔画粗度 40 乃至 80 px**。

度量: w_max = distance_transform_edt(ink_mask).max() * 2   —— 最粗单笔的宽度(px)
      w_avg = ink_area / skeleton_length                    —— 平均笔画宽度(px)
判据: dirty = (w_max > TH_W) or (ink > 0.5) or (med < 150) or (max_blob > 0.5)
输出: 各 TH_W 下的剩余量; 并把逐图指标缓存到 /tmp/scan_metrics.npz 供 apply 复用。
用法: python tools/rescan_clean.py [--rollback] [--th 40]
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(ROOT)

NOAUG = "assets/train_base_noaug.csv"
CACHE = "/tmp/scan_metrics.npz"
QUARS = ("data/_quarantine", "data/_quarantine2")


def rollback():
    n = 0
    for Q in QUARS:
        if not os.path.isdir(Q):
            continue
        for f in glob.glob(os.path.join(Q, "*", "*.png")):
            sd = os.path.basename(os.path.dirname(f))
            dst_dir = "data/imgs/base_sym" if sd == "base_sym" else f"data/imgs/{sd}"
            os.makedirs(dst_dir, exist_ok=True)
            try:
                shutil.move(f, os.path.join(dst_dir, os.path.basename(f)))
                n += 1
            except Exception as e:
                print("  fail", f, e)
    print(f"[rollback] restored {n} png")
    for Q in QUARS:
        if os.path.isdir(Q):
            for f in glob.glob(os.path.join(Q, "*", "*.png")):
                pass
            print(f"  {Q}: {len(glob.glob(os.path.join(Q,'*','*.png')))} 残留")


def probe(path):
    try:
        a = np.asarray(Image.open(path).convert("L"))
    except Exception:
        return (-1.0, -1.0, -1.0, -1.0, -1.0)
    m = a < 128
    n_px = a.size
    ink = float(m.mean())
    med = float(np.median(a))
    if m.any():
        w_max = float(distance_transform_edt(m).max()) * 2.0
        lab, _ = label(m)
        sizes = np.bincount(lab.ravel())[1:]
        max_blob = float(sizes.max()) / n_px
    else:
        w_max, max_blob = 0.0, 0.0
    # 反色检测: 正常图**亮部(白底)必然接触图像边缘**; 反色/黑底图的亮部
    # 是"被暗部包住"的孤立区域 (如反色的"一": 黑底白横), 边缘几乎全暗。
    e = (a[0, :] > 128).mean() + (a[-1, :] > 128).mean() + \
        (a[:, 0] > 128).mean() + (a[:, -1] > 128).mean()
    edge_bright = float(e) / 4.0
    # inner_ratio: **不接触图像边界**的亮部面积占全部亮部的比例。
    #   正常图: 白底连通到边缘 -> 接近 0 (字内封闭空隙占比很小)
    #   反色图: 白字/白横条是被黑包住的孤岛 -> 接近 1
    bright = a > 128
    inner = 0.0
    tot_b = int(bright.sum())
    if tot_b:
        lb, nb = label(bright)
        sizes = np.bincount(lb.ravel(), minlength=nb + 1)
        border = set(lb[0, :].tolist()) | set(lb[-1, :].tolist()) | \
                 set(lb[:, 0].tolist()) | set(lb[:, -1].tolist())
        border.discard(0)
        inner_area = tot_b - sum(int(sizes[l]) for l in border)
        inner = float(inner_area) / tot_b
    return (w_max, ink, med, max_blob, edge_bright, inner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--th", type=float, default=40.0)
    a = ap.parse_args()

    if a.rollback:
        rollback()

    rows = list(csv.DictReader(open(NOAUG, encoding="utf-8")))
    paths = [r["image_path"] for r in rows]
    miss = sum(1 for p in paths if not os.path.exists(p))
    print(f"[scan] {len(paths)} 张, 缺失 {miss}", flush=True)

    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if len(d["w_max"]) == len(paths):
            metrics = d
            print("[scan] 复用缓存", CACHE)
        else:
            metrics = None
    else:
        metrics = None

    if metrics is None:
        t0 = time.time()
        # ⚠ 必须保序: imap_unordered 的返回顺序与输入不一致, 会让指标与图片错位
        #   (2026-09-15 踩坑: 导致"干净"判定张冠李戴, clean 里混入 130px 粗笔画)。
        with mp.Pool(40) as pool:
            res = pool.map(probe, paths, chunksize=64)
        print(f"  scanned {len(res)} ({time.time()-t0:.0f}s)", flush=True)
        arr = np.array(res, dtype=np.float32)
        np.savez(CACHE, w_max=arr[:, 0], ink=arr[:, 1], med=arr[:, 2],
                 blob=arr[:, 3], edge_bright=arr[:, 4], inner=arr[:, 5])
        metrics = np.load(CACHE)
        print(f"[scan] done {time.time()-t0:.0f}s -> {CACHE}")

    w, ink, med, blob = metrics["w_max"], metrics["ink"], metrics["med"], metrics["blob"]
    print(f"\n  w_max(最粗单笔 px): p50={np.percentile(w,50):.1f} "
          f"p75={np.percentile(w,75):.1f} p90={np.percentile(w,90):.1f} "
          f"p99={np.percentile(w,99):.1f} max={w.max():.1f}")
    print(f"  ink: p50={np.percentile(ink,50):.3f} p99={np.percentile(ink,99):.3f}")
    print(f"  blob: p50={np.percentile(blob,50):.3f} p99={np.percentile(blob,99):.3f}")

    other = (ink > 0.5) | (med < 150) | (blob > 0.5)
    print(f"\n  非笔画类脏图 (ink>0.5 或 med<150 或 blob>0.5): {int(other.sum())}")
    print("\n  === 不同 [最粗单笔宽度] 阈值下的结果 ===")
    for th in (26, 30, 35, 40, 50, 60, 80, 100):
        d = other | (w > th)
        print(f"    w_max>{th:3d}px -> dirty {int(d.sum()):6d}  clean {int((~d).sum()):6d}")


if __name__ == "__main__":
    main()
