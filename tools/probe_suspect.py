# -*- coding: utf-8 -*-
"""probe_suspect.py — 揪出 clean 组里"最可疑"的图并实测其宽度, 定位漏剔原因.

可疑定义: clean 组 (w_max<=th) 中 ink 或 blob 偏高者 —— 反色/黑块的特征。
对每张重新实测 w_dt / w_ero / blob / ink / 边缘暗度, 并导出原尺寸图供目测。
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label, generate_binary_structure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ST = generate_binary_structure(2, 2)
CACHE = "/tmp/scan_metrics.npz"
NOAUG = "assets/train_base_noaug.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(NOAUG, encoding="utf-8")))
    paths = [r["image_path"] for r in rows]
    d = np.load(CACHE)
    w, ink, med, blob, eb = (d["w_max"], d["ink"], d["med"], d["blob"],
                             d["edge_bright"])
    clean = np.nonzero(w <= a.th)[0]
    # 可疑度 = blob 大 或 边缘暗 (反色特征)
    score = blob[clean] * 2.0 + (1.0 - eb[clean])
    order = clean[np.argsort(-score)]
    print(f"[probe] th={a.th}  clean={len(clean)}  取最可疑 {a.n} 张")

    out = "/root/Workspace/xy/DiT/_otout_suspect"
    os.makedirs(out, exist_ok=True)
    lines = []
    for i, k in enumerate(order[:a.n]):
        p = paths[k]
        try:
            arr = np.asarray(Image.open(p).convert("L"))
        except Exception:
            continue
        m = arr < 128
        w_dt = float(distance_transform_edt(m).max()) * 2.0 if m.any() else 0.0
        mm = m.copy()
        c = 0
        while mm.any() and c < 300:
            mm = binary_erosion(mm, ST)
            c += 1
        w_ero = c * 2.0
        lb, _ = label(m)
        sz = np.bincount(lb.ravel())[1:]
        b2 = float(sz.max()) / arr.size if sz.size else 0.0
        # 反色: 亮部不接触边框的比例
        br = arr > 128
        inner = 0.0
        if br.any():
            lb2, nb2 = label(br)
            s2 = np.bincount(lb2.ravel(), minlength=nb2 + 1)
            bd = set(lb2[0, :].tolist()) | set(lb2[-1, :].tolist()) | \
                 set(lb2[:, 0].tolist()) | set(lb2[:, -1].tolist())
            bd.discard(0)
            inner = float(br.sum() - sum(int(s2[x]) for x in bd)) / br.sum()
        lines.append((p, r["character"] if (r := rows[k]) else "", w_dt, w_ero, b2,
                      float(m.mean()), inner))
        Image.fromarray(arr).save(
            f"{out}/{i:02d}_wdt{w_dt:.0f}_blob{b2:.2f}_inner{inner:.2f}_"
            f"{rows[k].get('character','')}.png")

    print(f"  {'img':40s} {'字':2s} {'w_dt':>6s} {'w_ero':>6s} {'blob':>6s} "
          f"{'ink':>6s} {'inner':>6s}")
    for p, ch, wd, we, b2, ik, inn in lines:
        print(f"  {p[-38:]:40s} {ch:2s} {wd:6.1f} {we:6.1f} {b2:6.3f} {ik:6.3f} {inn:6.3f}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
