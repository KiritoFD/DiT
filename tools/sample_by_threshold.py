# -*- coding: utf-8 -*-
"""sample_by_threshold.py — 导出 55px / 60px 阈值分别剔掉的图, 供目视对比.

组 (均按 w_max 降序, 最粗排前面):
  kill60      : w_max > 60        (60px 判据剔除)
  diff_55_60  : 55 < w_max <= 60   (55px 会剔但 60px 保留 —— 边界区)
  diff_50_55  : 50 < w_max <= 55
  kept_45_50  : 45 < w_max <= 50   (都保留的, 作对照)
"""
import csv
import multiprocessing as mp
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KEEP = ("final_imgs_fame_v8", "calli_tongji_imgs")
OUT = "/root/Workspace/xy/DiT/_otout_thresh"
os.makedirs(OUT, exist_ok=True)


def probe(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return -1.0
    m = a < 128
    return float(distance_transform_edt(m).max()) * 2.0 if m.any() else 0.0


def montage(items, out, cols=8, cell=110):
    n = min(len(items), cols * 6)
    rows = max((n + cols - 1) // cols, 1)
    cv = Image.new("RGB", (cols * cell, rows * cell), (180, 30, 30))
    for i, (p, w) in enumerate(items[:n]):
        try:
            im = Image.open(p).convert("L").resize((cell, cell))
            cv.paste(im.convert("RGB"), ((i % cols) * cell, (i // cols) * cell))
        except Exception:
            pass
    cv.save(out)
    print(f"  -> {out}  ({min(len(items),cols*6)} 张)")


def main():
    rows = [r for r in csv.DictReader(
        open("assets/train_base_noaug.csv", encoding="utf-8"))
        if any(s in r["image_path"] for s in KEEP)]
    print(f"[thresh] {len(rows)} 张", flush=True)
    with mp.Pool(40) as pool:
        w = pool.map(probe, [r["image_path"] for r in rows], chunksize=256)

    recs = [(r["image_path"], float(x)) for r, x in zip(rows, w)]
    groups = {
        "kill60": [t for t in recs if t[1] > 60],
        "diff_55_60": [t for t in recs if 55 < t[1] <= 60],
        "diff_50_55": [t for t in recs if 50 < t[1] <= 55],
        "kept_45_50": [t for t in recs if 45 < t[1] <= 50],
    }
    for g, v in groups.items():
        v.sort(key=lambda x: -x[1])
        print(f"  {g:12s} {len(v):6d}  (w 范围 "
              f"{v[-1][1]:.1f}~{v[0][1]:.1f})" if v else f"  {g:12s} 0")
        montage(v, f"{OUT}/{g}.png")

    # 每组的 8 个代表 (最大/中位/最小各若干) 单独存原尺寸, 便于细看
    for g, v in groups.items():
        if not v:
            continue
        step = max(1, len(v) // 8)
        for i, (p, wv) in enumerate(v[::step][:8]):
            try:
                Image.open(p).convert("L").save(f"{OUT}/{g}_{i:02d}_w{wv:.0f}.png")
            except Exception:
                pass
    print(f"  -> {OUT} (组图 + 每组8张原尺寸)")


if __name__ == "__main__":
    main()
