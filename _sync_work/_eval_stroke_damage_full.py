# -*- coding: utf-8 -*-
"""_eval_stroke_damage_full.py — 全量笔画级损伤评估 (远程跑).

对比 data/imgs/final_imgs_fame_v8 (原始) vs data/imgs/final_imgs_fame_v9_REJECTED (v3.7 去噪后),
用笔画级指标判定 v3.7 是否伤字形:

  skeleton_keep : v8 骨架像素在去噪后仍为墨的比例 (<0.95 = 笔画断裂/消失)
  half_width    : 骨架处局部半宽 (distance_transform), 反映笔画被削细多少

用法(远程):
  /opt/conda/bin/python _sync_work/_eval_stroke_damage_full.py --workers 8
"""
import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

A_ROOT = "data/imgs/final_imgs_fame_v8"
B_ROOT = "data/imgs/final_imgs_fame_v9_REJECTED"
CSV = "assets/train_fame_clean_v8.csv"
OUT = "assets/stroke_damage_v9.csv"


def to_ink(a):
    ink = a < 128
    if ink.mean() > 0.5:
        ink = ~ink
    return ink


def work(rel):
    iid = os.path.splitext(os.path.basename(rel))[0]
    try:
        a8 = np.asarray(Image.open(os.path.join(A_ROOT, rel)).convert("L"))
        a9 = np.asarray(Image.open(os.path.join(B_ROOT, rel)).convert("L"))
    except Exception as e:
        return {"img_id": iid, "ok": 0, "err": str(e)[:50]}
    ink8 = to_ink(a8)
    ink9 = to_ink(a9)
    if ink9.shape != ink8.shape:
        ink9 = np.asarray(Image.fromarray((ink9 * 255).astype(np.uint8))
                          .resize((ink8.shape[1], ink8.shape[0]),
                                  Image.NEAREST)) > 0
    n8, n9 = int(ink8.sum()), int(ink9.sum())
    if n8 == 0:
        return {"img_id": iid, "ok": 0, "err": "empty"}
    sk8 = skeletonize(ink8)
    if not sk8.any():
        return {"img_id": iid, "ok": 0, "err": "noskel"}
    keep = float(ink9[sk8].mean())
    dt8 = ndimage.distance_transform_edt(ink8)
    dt9 = ndimage.distance_transform_edt(ink9)
    hw8 = float(dt8[sk8].mean())
    hw9 = float(dt9[sk8].mean())
    return {
        "img_id": iid, "ok": 1, "err": "",
        "ink_delta": round((n9 - n8) / max(n8, 1), 5),
        "skeleton_keep": round(keep, 5),
        "half_w_before": round(hw8, 3),
        "half_w_after": round(hw9, 3),
        "width_change": round((hw9 - hw8) / max(hw8, 1e-6), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rels = sorted(set(os.path.basename(r["image_path"]) for r in rows))
    if args.limit:
        rels = rels[:args.limit]
    print(f"[eval] {len(rels)} images: {A_ROOT} vs {B_ROOT}", flush=True)

    FIELDS = ["img_id", "ok", "err", "ink_delta", "skeleton_keep",
              "half_w_before", "half_w_after", "width_change"]
    out = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, rels, chunksize=64)):
            out.append(r)
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(rels)}", flush=True)

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"[done] {OUT} ({len(out)} rows)", flush=True)

    ok = [r for r in out if r.get("ok") == 1]
    if not ok:
        return
    n = len(ok)
    keep = np.array([r["skeleton_keep"] for r in ok], dtype=float)
    wid = np.array([r["width_change"] for r in ok], dtype=float)
    hw = np.array([r["half_w_before"] for r in ok], dtype=float)
    print(f"\n=== 全量笔画级评估 ({n} 图) ===")
    print(f"  skeleton_keep mean  = {keep.mean():.4f}")
    print(f"  skeleton_keep<0.98  = {int((keep<0.98).sum())} "
          f"({100*(keep<0.98).mean():.2f}%)")
    print(f"  skeleton_keep<0.95  = {int((keep<0.95).sum())} "
          f"({100*(keep<0.95).mean():.2f}%)  ← 笔画断裂/消失")
    print(f"  skeleton_keep<0.90  = {int((keep<0.90).sum())} "
          f"({100*(keep<0.90).mean():.2f}%)  ← 严重")
    print(f"  width_change mean   = {100*wid.mean():.2f}%")
    print(f"  width_change<-20%   = {int((wid<-0.20).sum())} "
          f"({100*(wid<-0.20).mean():.2f}%)")
    print(f"\n  按笔画粗细分组 (半宽 before):")
    for lo, hi, name in [(0, 2.5, "细 (<2.5px)"), (2.5, 4.0, "中 (2.5-4px)"),
                         (4.0, 6.0, "粗 (4-6px)"), (6.0, 999, "极粗 (>6px)")]:
        m = (hw >= lo) & (hw < hi)
        if m.sum() == 0:
            continue
        print(f"    {name:<14} n={int(m.sum()):>6} "
              f"keep={keep[m].mean():.4f} "
              f"keep<0.95={100*(keep[m]<0.95).mean():>6.2f}% "
              f"宽度变化={100*wid[m].mean():>7.2f}%")


if __name__ == "__main__":
    main()
