# -*- coding: utf-8 -*-
"""audit_clean_v1.py — 复核 clean_v1 的图, 找出"空白图"和"大黑块图".

55px 判据只度量"最粗处宽度", 对**扁黑块**(宽>高, 如 200x40)标不出,
对**空白图**更是完全不管。这里补两条:
  E1 blank : ink < 0.01            (几乎无笔画)
  E2 blob  : 最大暗连通域面积占比 > 0.30  (整块黑)
同时给出 55px 判据的复核值, 便于对照。
"""
import csv
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV = sys.argv[1] if len(sys.argv) > 1 else "assets/train_clean_v1.csv"


def probe(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (p, -1.0, -1.0, -1.0, "read_error")
    m = a < 128
    ink = float(m.mean())
    if m.any():
        w = float(distance_transform_edt(m).max()) * 2.0
        lb, _ = label(m)
        sz = np.bincount(lb.ravel())[1:]
        blob = float(sz.max()) / a.size
    else:
        w, blob = 0.0, 0.0
    bad = []
    if ink < 0.01:
        bad.append("E1_blank")
    if blob > 0.30:
        bad.append("E2_blob")
    return (p, ink, w, blob, "+".join(bad))


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    paths = [r["image_path"] for r in rows]
    print(f"[audit] {CSV}: {len(paths)} 行", flush=True)
    with mp.Pool(40) as pool:
        res = pool.map(probe, paths, chunksize=128)

    c = Counter()
    for r in res:
        c[r[-1] or "ok"] += 1
    print("\n=== 结果 ===")
    for k, v in c.most_common():
        print(f"  {k:16s} {v}")
    bad = [r for r in res if r[-1]]
    print(f"\n  坏图合计 {len(bad)} / {len(res)} ({100*len(bad)/len(res):.2f}%)")

    print("\n  样例:")
    for r in bad[:15]:
        print(f"    ink={r[1]:.4f} w={r[2]:6.1f} blob={r[3]:.3f} {r[4]:12s} {r[0][-40:]}")

    inks = np.array([r[1] for r in res])
    blobs = np.array([r[3] for r in res])
    print(f"\n  ink:  p0.1={np.percentile(inks,0.1):.4f} p1={np.percentile(inks,1):.4f} "
          f"p50={np.percentile(inks,50):.4f}")
    print(f"  blob: p50={np.percentile(blobs,50):.3f} p90={np.percentile(blobs,90):.3f} "
          f"p99={np.percentile(blobs,99):.3f} max={blobs.max():.3f}")
    for t in (0.2, 0.25, 0.3, 0.4, 0.5):
        print(f"    blob>{t}: {int((blobs>t).sum())}")

    with open("/tmp/clean_v1_bad.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "ink", "w_max", "blob", "reason"])
        for r in bad:
            w.writerow(r)
    print(f"\n  -> /tmp/clean_v1_bad.csv")


if __name__ == "__main__":
    main()
