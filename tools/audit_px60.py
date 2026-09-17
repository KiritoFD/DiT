# -*- coding: utf-8 -*-
"""audit_px60.py — 复核 px60 判据的漏网: 找"亮部不接触图像边界"的图.

正常白底书法: 亮部(背景)必然**连通到图像边界** -> inner≈0
反色/黑底/黑包络: 亮部是被暗部包住的孤岛 -> inner≈1
  (如反色的"一": 黑底 + 白横; 早期 fix_polarity 把白 pad 反成黑框的那批)

纯 w_max>60 抓不到"黑**框**"形态 —— 因为 DT 量的是框的**厚度**而非包络整体尺寸。
本脚本只做**诊断**(不剔除), 给出漏网数量与全尺寸样例, 供决策。
"""
import csv
import multiprocessing as mp
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KEEP_SRC = ("final_imgs_fame_v8", "calli_tongji_imgs")
NAME = "fame-kxl-tj-px60"
CACHE = f"/tmp/{NAME}_wmax.npz"
OUT = f"/root/Workspace/xy/DiT/_otout_{NAME}_audit"
TH = 60.0


def probe(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (-1.0, -1.0, -1.0)
    m = a < 128
    ink = float(m.mean())
    w = float(distance_transform_edt(m).max()) * 2.0 if m.any() else 0.0
    bright = a > 128
    inner = 0.0
    tot = int(bright.sum())
    if tot:
        lb, nb = label(bright)
        sizes = np.bincount(lb.ravel(), minlength=nb + 1)
        border = set(lb[0, :].tolist()) | set(lb[-1, :].tolist()) | \
                 set(lb[:, 0].tolist()) | set(lb[:, -1].tolist())
        border.discard(0)
        inner = float(tot - sum(int(sizes[l]) for l in border)) / tot
    return (w, ink, inner)


def montage(paths, out, cols=6, cell=190):
    n = min(len(paths), cols * 3)
    rows = max((n + cols - 1) // cols, 1)
    cv = Image.new("RGB", (cols * cell, rows * cell), (150, 30, 30))
    for i, p in enumerate(paths[:n]):
        try:
            cv.paste(Image.open(p).convert("L").resize((cell, cell)).convert("RGB"),
                     ((i % cols) * cell, (i // cols) * cell))
        except Exception:
            pass
    cv.save(out)
    print(f"  -> {out} ({n} 张)")


def main():
    rows = [r for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8"))
            if any(s in r["image_path"] for s in KEEP_SRC)]
    w = np.load(CACHE)["w"]
    assert len(w) == len(rows)
    keep = [i for i in range(len(rows)) if w[i] <= TH]
    print(f"[audit] {NAME}: 总 {len(rows)}, keep(w<={TH:.0f}px) {len(keep)}")

    with mp.Pool(40) as pool:
        res = pool.map(probe, [rows[i]["image_path"] for i in keep], chunksize=128)
    ww = np.array([r[0] for r in res])
    ink = np.array([r[1] for r in res])
    inn = np.array([r[2] for r in res])
    print(f"\n  keep 组 w_max: p50={np.percentile(ww,50):.1f} p99={np.percentile(ww,99):.1f} max={ww.max():.1f}")
    print(f"  keep 组 inner: p50={np.percentile(inn,50):.4f} p99={np.percentile(inn,99):.4f} max={inn.max():.4f}")
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        n = int((inn > t).sum())
        print(f"    inner>{t:.2f}: {n:6d}")

    order = np.argsort(-inn)
    print(f"\n  最可疑 (inner 最大) 12 张:")
    for k in order[:12]:
        print(f"    inner={inn[k]:.4f} w={ww[k]:6.1f} ink={ink[k]:.3f} "
              f"char={rows[keep[k]].get('character','')} "
              f"{rows[keep[k]]['image_path']}")

    os.makedirs(OUT, exist_ok=True)
    montage([rows[keep[k]]["image_path"] for k in order[:18]], f"{OUT}/top_inner.png")
    # w 最接近 60 的 (判据边界)
    o2 = np.argsort(-ww)
    montage([rows[keep[k]]["image_path"] for k in o2[:18]], f"{OUT}/top_w.png")


if __name__ == "__main__":
    main()
