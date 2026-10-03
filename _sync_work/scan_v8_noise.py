# -*- coding: utf-8 -*-
"""scan_v8_noise.py — 扩展噪点指标全量扫描 (与 scan_image_pollution.py 同口径 + 新增指标).

同口径参数(保证可与 scan_train_pollution.csv 直接对比):
  EDGE_PX=8, SMALL_FRAC=0.0005, MID_FRAC=0.01, BAR_THR=0.90

新增指标(针对 v8 清洗后残留的装裱黑边/灰度噪点):
  top_ratio / bot_ratio / left_ratio / right_ratio : 单边 8px 条带前景占比
  gray_mid_ratio : 中间灰度像素占比 (30<v<200) —— 抗锯齿/未二值化的脏背景
  lap_var        : 拉普拉斯方差 —— 高频噪点(椒盐/锯齿)强度

用法(远程):
  /opt/conda/bin/python _sync_work/scan_v8_noise.py \
      --csv assets/train_fame_clean_v8.csv --img-root data/imgs/final_imgs_fame_v8 \
      --out assets/scan_v8_noise.csv --workers 16
"""
import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EDGE_PX = 8
SMALL_FRAC = 0.0005
MID_FRAC = 0.01
BAR_THR = 0.90


def analyze(path):
    from PIL import Image
    from scipy import ndimage
    try:
        a = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception:
        return None
    H, W = a.shape
    N = float(H * W)
    ink = a < 128
    n_ink = int(ink.sum())
    r = {"img_id": os.path.splitext(os.path.basename(path))[0],
         "H": H, "W": W, "ink_ratio": round(n_ink / N, 6), "n_ink": n_ink}

    ZERO = {k: 0 for k in ["n_cc", "main_frac", "small_cc_n",
                           "small_cc_area_ratio", "mid_cc_n",
                           "foreign_area_ratio", "edge_ink_ratio",
                           "top_ratio", "bot_ratio", "left_ratio",
                           "right_ratio", "border_bar", "edge_blob_area",
                           "inverted", "gray_mid_ratio", "lap_var"]}
    if n_ink == 0:
        r.update(ZERO)
        return r

    lab, nlab = ndimage.label(ink)
    if nlab == 0:
        r.update(ZERO)
        return r

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main_label = int(sizes.argmax())
    main_area = int(sizes[main_label])
    foreign = ink & (lab != main_label)

    small_cc_n = 0
    small_area = 0
    mid_cc_n = 0
    for lb in range(1, nlab + 1):
        if lb == main_label:
            continue
        ar = int(sizes[lb])
        if ar < SMALL_FRAC * N:
            small_cc_n += 1
            small_area += ar
        elif ar < MID_FRAC * N:
            mid_cc_n += 1

    b = EDGE_PX
    bm = np.zeros((H, W), dtype=bool)
    bm[:b, :] = True
    bm[-b:, :] = True
    bm[:, :b] = True
    bm[:, -b:] = True
    edge_ratio = (ink & bm).sum() / float(bm.sum())
    top_ratio = float(ink[:b, :].mean())
    bot_ratio = float(ink[-b:, :].mean())
    left_ratio = float(ink[:, :b].mean())
    right_ratio = float(ink[:, -b:].mean())

    bar = 0
    for strip in (ink[:b, :], ink[-b:, :], ink[:, :b], ink[:, -b:]):
        if strip.size and (strip.sum() / float(strip.size)) > BAR_THR:
            bar = 1
            break

    edge_blob_area = 0
    if foreign.any():
        for lb in range(1, nlab + 1):
            if lb == main_label:
                continue
            m = (lab == lb)
            if (m & bm).any() and int(sizes[lb]) >= MID_FRAC * N:
                edge_blob_area += int(sizes[lb])

    gray_mid_ratio = float(((a > 30) & (a < 200)).sum()) / N
    lap = ndimage.laplace(a.astype(np.float32))
    lap_var = float(lap.var())

    r.update({
        "n_cc": int(nlab),
        "main_frac": round(main_area / float(n_ink), 6),
        "small_cc_n": small_cc_n,
        "small_cc_area_ratio": round(small_area / N, 6),
        "mid_cc_n": mid_cc_n,
        "foreign_area_ratio": round(int(foreign.sum()) / N, 6),
        "edge_ink_ratio": round(edge_ratio, 6),
        "top_ratio": round(top_ratio, 6),
        "bot_ratio": round(bot_ratio, 6),
        "left_ratio": round(left_ratio, 6),
        "right_ratio": round(right_ratio, 6),
        "border_bar": bar,
        "edge_blob_area": edge_blob_area,
        "inverted": 1 if (n_ink / N) > 0.5 else 0,
        "gray_mid_ratio": round(gray_mid_ratio, 6),
        "lap_var": round(lap_var, 2),
    })
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_fame_clean_v8.csv")
    ap.add_argument("--img-root", default="data/imgs/final_imgs_fame_v8")
    ap.add_argument("--out", default="assets/scan_v8_noise.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    paths = []
    for r in rows:
        p = r["image_path"]
        if not os.path.isabs(p):
            p = os.path.join(args.img_root, os.path.basename(p))
        if os.path.isfile(p):
            paths.append(p)
    paths = sorted(set(paths))
    if args.limit:
        paths = paths[:args.limit]
    print(f"[scan] {len(paths)} images -> {args.out}", flush=True)

    FIELDS = ["img_id", "H", "W", "ink_ratio", "n_ink", "n_cc", "main_frac",
              "small_cc_n", "small_cc_area_ratio", "mid_cc_n",
              "foreign_area_ratio", "edge_ink_ratio", "top_ratio",
              "bot_ratio", "left_ratio", "right_ratio", "border_bar",
              "edge_blob_area", "inverted", "gray_mid_ratio", "lap_var"]

    out = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(analyze, paths, chunksize=64)):
            if res is not None:
                out.append(res)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(paths)}", flush=True)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"[done] wrote {len(out)} rows -> {args.out}", flush=True)

    # 汇总(与本地 scan_train_pollution.csv 同口径可比)
    if out:
        n = len(out)
        def pct(k, pred):
            return f"{sum(1 for r in out if pred(r[k]))} " \
                   f"({100*sum(1 for r in out if pred(r[k]))/n:.2f}%)"
        F = float
        print("\n=== 汇总 ===")
        print(f"  总图数: {n}")
        print(f"  small_cc_n>0:  {pct('small_cc_n', lambda v: int(v)>0)}")
        print(f"  small_cc_n>3:  {pct('small_cc_n', lambda v: int(v)>3)}")
        print(f"  small_cc_n>5:  {pct('small_cc_n', lambda v: int(v)>5)}")
        print(f"  small_cc_n>10: {pct('small_cc_n', lambda v: int(v)>10)}")
        print(f"  small_a>0.0005:{pct('small_cc_area_ratio', lambda v: F(v)>0.0005)}")
        print(f"  small_a>0.005: {pct('small_cc_area_ratio', lambda v: F(v)>0.005)}")
        print(f"  edge>0.05:     {pct('edge_ink_ratio', lambda v: F(v)>0.05)}")
        print(f"  edge>0.15:     {pct('edge_ink_ratio', lambda v: F(v)>0.15)}")
        print(f"  border_bar=1:  {pct('border_bar', lambda v: int(v)==1)}")
        print(f"  edge_blob>0:   {pct('edge_blob_area', lambda v: int(v)>0)}")
        print(f"  inverted=1:    {pct('inverted', lambda v: int(v)==1)}")
        for side in ("top_ratio", "bot_ratio", "left_ratio", "right_ratio"):
            print(f"  {side}>0.30:  {pct(side, lambda v: F(v)>0.30)}")
        # 综合
        un = sum(1 for r in out
                 if int(r['small_cc_n'])>3 or float(r['edge_ink_ratio'])>0.15
                 or int(r['border_bar'])==1 or int(r['inverted'])==1
                 or int(r['edge_blob_area'])>0)
        print(f"  综合有噪点(sc_n>3 OR edge>0.15 OR bar OR inv OR blob>0): "
              f"{un} ({100*un/n:.2f}%)")


if __name__ == "__main__":
    main()
