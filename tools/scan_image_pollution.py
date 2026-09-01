# -*- coding: utf-8 -*-
"""scan_image_pollution.py — 全量扫描 GT 图像污染现状，量化三类问题。

扫描对象: 训练集 (train_fame.csv) + 评测集 (eval_fame_strict.csv) 全量图。
对每个图计算一组"污染指标"，输出逐图 CSV 供后续分析/分级/方案评估。

三类污染定义
-----------
1) 噪点 (noise): 与主连通域(主字)不连通的孤立小墨点。
   - 指标: n_cc (连通域总数), small_cc_n (面积<阈值的非主连通域数),
           small_cc_area_ratio (这些小点总面积/全图)
2) 脏污 (dirt): 较大的非主连通域斑块(非噪点级)，或贴附在主字上的异常墨块。
   - 指标: mid_cc_n (中等大小非主连通域数), foreign_area_ratio (非主前景总面积/全图)
3) 连边大片黑 (edge_blob): 边界环带内的大片前景(黑框/裁切残片/实心带)。
   - 指标: edge_ink_ratio (边界环带内前景像素/环带面积),
           border_bar (某条边是否近似实心黑条),
           edge_blob_area (边界连通域且面积较大的前景面积)

其他形态指标
-----------
- ink_ratio: 全图前景(墨)比例 (用于识别反相图: >0.5 疑似白字黑底)
- main_frac: 主连通域面积/全图前景面积 (主字占比, 低=碎片多)
- n_cc: 连通域总数

用法:
  python tools/scan_image_pollution.py --csv 5script/train_fame.csv \
      --out scan_train.csv [--workers 32]
"""
import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EDGE_PX = 8          # 边界环带宽度
SMALL_FRAC = 0.0005  # 小连通域阈值 (占全图面积) = 0.05%
MID_FRAC = 0.01      # 中等连通域上限 (占全图面积) = 1%
BAR_THR = 0.90       # 某条边连续前景占比超过此值视为"实心黑条"


def analyze(path):
    """返回单图的污染指标 dict。"""
    from PIL import Image
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return None
    a = np.asarray(img, dtype=np.uint8)
    H, W = a.shape
    N = float(H * W)
    ink = a < 128
    n_ink = int(ink.sum())
    r = {
        "img_id": os.path.splitext(os.path.basename(path))[0],
        "H": H, "W": W,
        "ink_ratio": round(n_ink / N, 6),
        "n_ink": n_ink,
    }
    if n_ink == 0:
        r.update({"n_cc": 0, "main_frac": 0.0, "small_cc_n": 0,
                  "small_cc_area_ratio": 0.0, "mid_cc_n": 0,
                  "foreign_area_ratio": 0.0, "edge_ink_ratio": 0.0,
                  "border_bar": 0, "edge_blob_area": 0, "inverted": 0})
        return r

    try:
        from scipy import ndimage
        lab, nlab = ndimage.label(ink)
    except Exception:
        lab, nlab = None, 0
    if lab is None or nlab == 0:
        r.update({"n_cc": 0, "main_frac": 0.0, "small_cc_n": 0,
                  "small_cc_area_ratio": 0.0, "mid_cc_n": 0,
                  "foreign_area_ratio": 0.0, "edge_ink_ratio": 0.0,
                  "border_bar": 0, "edge_blob_area": 0, "inverted": 0})
        return r

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main_label = int(sizes.argmax())
    main_area = int(sizes[main_label])
    foreign = ink & (lab != main_label)

    # 小/中等连通域
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

    # 边界环带
    b = EDGE_PX
    bm = np.zeros((H, W), dtype=bool)
    bm[:b, :] = True
    bm[-b:, :] = True
    bm[:, :b] = True
    bm[:, -b:] = True
    edge_ink = int((ink & bm).sum())
    edge_ratio = edge_ink / float(bm.sum())

    # 实心黑条: 检查四条边的前 b 行/列，整体前景占比
    bar = 0
    for strip in (ink[:b, :], ink[-b:, :], ink[:, :b], ink[:, -b:]):
        if strip.size and (strip.sum() / float(strip.size)) > BAR_THR:
            bar = 1
            break

    # 边界大片黑: 与边界接触的连通域中, 面积 > MID_FRAC 的那些的总面积
    edge_blob_area = 0
    if foreign.any():
        touch = set()
        for lb in range(1, nlab + 1):
            if lb == main_label:
                continue
            m = (lab == lb)
            # 是否接触边界环带
            if (m & bm).any() and int(sizes[lb]) >= MID_FRAC * N:
                touch.add(lb)
        for lb in touch:
            edge_blob_area += int(sizes[lb])

    r.update({
        "n_cc": int(nlab),
        "main_frac": round(main_area / float(n_ink), 6),
        "small_cc_n": small_cc_n,
        "small_cc_area_ratio": round(small_area / N, 6),
        "mid_cc_n": mid_cc_n,
        "foreign_area_ratio": round(int(foreign.sum()) / N, 6),
        "edge_ink_ratio": round(edge_ratio, 6),
        "border_bar": bar,
        "edge_blob_area": edge_blob_area,
        "inverted": 1 if (n_ink / N) > 0.5 else 0,
    })
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_fame.csv")
    ap.add_argument("--img-root", default="final_imgs_256")
    ap.add_argument("--out", default="scan_train.csv")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    paths = []
    for r in rows:
        p = r["image_path"]
        if not os.path.isabs(p):
            p = os.path.join(args.img_root, os.path.basename(p)) \
                if not os.path.isfile(p) else p
        if os.path.isfile(p):
            paths.append(p)
    if args.limit:
        paths = paths[:args.limit]
    print(f"[scan] {len(paths)} images -> {args.out}", flush=True)

    FIELDS = ["img_id", "H", "W", "ink_ratio", "n_ink", "n_cc", "main_frac",
              "small_cc_n", "small_cc_area_ratio", "mid_cc_n",
              "foreign_area_ratio", "edge_ink_ratio", "border_bar",
              "edge_blob_area", "inverted"]
    out = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(analyze, paths, chunksize=32)):
            if res is not None:
                out.append(res)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(paths)}", flush=True)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"[done] wrote {len(out)} rows -> {args.out}")

    # 汇总统计
    if out:
        arr = {k: np.array([r[k] for r in out], dtype=float) for k in FIELDS
               if k not in ("img_id",)}
        print("\n=== 汇总 ===")
        print(f"  总图数: {len(out)}")
        for k in ["ink_ratio", "n_cc", "main_frac", "small_cc_n",
                  "small_cc_area_ratio", "mid_cc_n", "foreign_area_ratio",
                  "edge_ink_ratio"]:
            v = arr[k]
            print(f"  {k:<22} mean={v.mean():.5f} median={np.median(v):.5f} "
                  f"p95={np.percentile(v,95):.5f} max={v.max():.5f}")
        print(f"  border_bar=1 (实心黑条): {int(arr['border_bar'].sum())} "
              f"({arr['border_bar'].mean()*100:.2f}%)")
        print(f"  inverted (疑似反相):   {int(arr['inverted'].sum())} "
              f"({arr['inverted'].mean()*100:.2f}%)")
        print(f"  small_cc_n>0 (有噪点):  {int((arr['small_cc_n']>0).sum())} "
              f"({(arr['small_cc_n']>0).mean()*100:.2f}%)")
        print(f"  edge_blob_area>0 (连边大片黑): {int((arr['edge_blob_area']>0).sum())} "
              f"({(arr['edge_blob_area']>0).mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
