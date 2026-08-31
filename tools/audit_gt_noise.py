# -*- coding: utf-8 -*-
"""audit_gt_noise.py — GT 图噪点审计: 生成训练 blacklist.

统计每张 GT 图的噪点指标, 标记离群样本:
  * ink_ratio    前景墨占比 (过高=糊, 过低=空)
  * small_blobs  孤立小墨斑数 (拓片墨渍/印泥/纸损)
  * edge_ink     边界 8px 前景占比 (裁切残片/边框)
  * main_frac    最大连通域占全部前景比例 (过低=碎屑多)

输出:
  gt_audit.csv        全量 per-image 指标
  gt_blacklist.csv    超阈值样本 (img_id, score, reasons)

用法 (远程 tmux, 纯 CPU 不占 GPU):
  /opt/conda/bin/python tools/audit_gt_noise.py --workers 16
"""
import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

IMG_ROOT = "final_imgs_256"
EDGE = 8
SMALL_AREA = 0.0005  # 小墨斑面积阈值 (占全图比例): <0.05%


def stats_one(path):
    try:
        a = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception:
        return None
    ink = a < 128
    n = ink.size
    ink_ratio = ink.sum() / n
    if ink_ratio == 0:
        return (path, 0.0, 0, 0.0, 1.0)
    # 连通域
    try:
        from scipy import ndimage
        lab, nlab = ndimage.label(ink)
        if nlab == 0:
            return (path, ink_ratio, 0, 0.0, 1.0)
        sizes = ndimage.sum(ink, lab, range(1, nlab + 1))
        main_frac = sizes.max() / ink.sum()
        small = int((sizes < n * SMALL_AREA).sum())
    except Exception:
        main_frac, small = 1.0, 0
    b = EDGE
    border = np.concatenate([ink[:b].ravel(), ink[-b:].ravel(),
                             ink[:, :b].ravel(), ink[:, -b:].ravel()])
    edge_ink = border.mean()
    return (path, float(ink_ratio), small, float(edge_ink), float(main_frac))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", default=IMG_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-csv", default="gt_audit.csv")
    ap.add_argument("--out-blacklist", default="gt_blacklist.csv")
    ap.add_argument("--ink-hi", type=float, default=0.45, help="墨占比上限")
    ap.add_argument("--ink-lo", type=float, default=0.01, help="墨占比下限")
    ap.add_argument("--blobs", type=int, default=30, help="小墨斑数上限")
    ap.add_argument("--edge", type=float, default=0.15, help="边缘墨占比上限")
    ap.add_argument("--main", type=float, default=0.55, help="主连通域占比下限")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.img_root) if f.endswith(".png"))
    paths = [os.path.join(args.img_root, f) for f in files]
    print(f"[audit] {len(paths)} images, workers={args.workers}", flush=True)

    with Pool(args.workers) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(stats_one, paths, chunksize=64)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(paths)}", flush=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "ink_ratio", "small_blobs", "edge_ink", "main_frac"])
        for path, ir, sb, ei, mf in rows:
            w.writerow([os.path.basename(path)[:-4], f"{ir:.4f}", sb, f"{ei:.4f}", f"{mf:.4f}"])
    print(f"[csv] {args.out_csv}: {len(rows)}")

    # 离群标记
    n_black = 0
    with open(args.out_blacklist, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "reasons"])
        for path, ir, sb, ei, mf in rows:
            reasons = []
            if ir > args.ink_hi:
                reasons.append(f"ink_hi:{ir:.3f}")
            if ir < args.ink_lo:
                reasons.append(f"ink_lo:{ir:.3f}")
            if sb > args.blobs:
                reasons.append(f"blobs:{sb}")
            if ei > args.edge:
                reasons.append(f"edge:{ei:.3f}")
            if mf < args.main:
                reasons.append(f"main_frac:{mf:.3f}")
            if reasons:
                w.writerow([os.path.basename(path)[:-4], ";".join(reasons)])
                n_black += 1
    print(f"[blacklist] {args.out_blacklist}: {n_black} ({n_black/len(rows)*100:.2f}%)")
    print("done.")


if __name__ == "__main__":
    main()
