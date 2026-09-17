# -*- coding: utf-8 -*-
"""sample_unicalli.py — 单独取样 UniCalli, 按判据分组导出 montage 供人工核验.

组:
  kept        : 通过分层清洗的
  drop_blob   : 大暗块 (黑底/墨团/拓片)
  drop_faint  : 过淡 (灰块/污渍/缺笔画)
  drop_bg     : 脏背景 (发灰发暗)
"""
import csv
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "/root/Workspace/xy/DiT/_otout_unicalli"
os.makedirs(OUT, exist_ok=True)
STRICT = dict(ink_lo=0.050, blob_hi=0.24, bg_lo=205.0)


def probe(task):
    k, p = task
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (k, 0.0, 0.0, 0.0, "err")
    m = a < 128
    ink = float(m.mean())
    blob = 0.0
    if m.any():
        lb, _ = label(m)
        sz = np.bincount(lb.ravel())[1:]
        blob = float(sz.max()) / a.size
    b = a[a > 128]
    bg = float(np.percentile(b, 50)) if b.size > 100 else 0.0
    rs = []
    if blob > STRICT["blob_hi"]:
        rs.append("blob")
    if ink < STRICT["ink_lo"]:
        rs.append("faint")
    if bg < STRICT["bg_lo"]:
        rs.append("bg")
    return (k, ink, blob, bg, "+".join(rs))


def montage(items, out, cols=8, cell=110):
    n = min(len(items), cols * 8)
    rows = max((n + cols - 1) // cols, 1)
    cv = Image.new("RGB", (cols * cell, rows * cell), (200, 30, 30))
    for i, (p, _, _, _, _) in enumerate(items[:n]):
        try:
            im = Image.open(p).convert("L").resize((cell, cell))
            cv.paste(im.convert("RGB"), ((i % cols) * cell, (i // cols) * cell))
        except Exception:
            pass
    cv.save(out)
    print(f"  -> {out}  ({min(len(items),cols*8)} 张)")


def main():
    rows = [r for r in csv.DictReader(
        open("assets/train_base_noaug.csv", encoding="utf-8"))
        if "unicalli_chars" in r["image_path"]]
    print(f"[sample] unicalli {len(rows)} 张")
    with mp.Pool(40) as pool:
        res = pool.map(probe, [(k, r["image_path"]) for k, r in enumerate(rows)],
                       chunksize=128)

    groups = {"kept": [], "drop_blob": [], "drop_faint": [], "drop_bg": [],
              "drop_multi": []}
    for (k, ink, blob, bg, reason), r in zip(res, rows):
        # 按**主因**归类 (blob 优先, 便于看清)
        if "blob" in reason:
            g = "drop_blob"
        elif "faint" in reason:
            g = "drop_faint"
        elif "bg" in reason:
            g = "drop_bg"
        else:
            g = "kept"
        groups[g].append((r["image_path"], ink, blob, bg, reason))

    print("\n  分组:")
    for g, v in groups.items():
        print(f"    {g:12s} {len(v):6d}")

    for g, v in groups.items():
        # 均匀抽样, 覆盖面更广
        step = max(1, len(v) // 64)
        sel = v[::step][:64]
        montage(sel, f"{OUT}/{g}.png")
        if sel:
            print(f"      样例: ink={sel[0][1]:.3f} blob={sel[0][2]:.3f} "
                  f"bg={sel[0][3]:.0f}")

    # 数值分布对照
    inks = np.array([x[1] for x in groups["kept"]] or [0])
    blobs = np.array([x[2] for x in groups["kept"]] or [0])
    bgs = np.array([x[3] for x in groups["kept"]] or [0])
    print(f"\n  kept 组: ink p50={np.percentile(inks,50):.3f} "
          f"blob p50={np.percentile(blobs,50):.3f} "
          f"bg p50={np.percentile(bgs,50):.0f}")


if __name__ == "__main__":
    main()
