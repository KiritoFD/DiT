# -*- coding: utf-8 -*-
"""aug_geo_audit.py — 增强的**定量 + 目视**审计（N≈200 张）

回答两个问题
------------
(1) **字有没有变**（拓扑是否保住）:
      IoU(墨迹) / 连通分量数 Δ（断笔!） / 封闭洞数 Δ（包围结构!） / 墨面积比 / 笔宽代理
(2) **有没有训练价值**:
      同图两次独立增强之间的 IoU（=1 说明"白增强"） / 墨面积分布展宽
(3) 附加（决定 REPA 缓存能否复用）:
      DINOv2 teacher patch 特征的余弦相似度（原图 vs 增强）—— **有权重时才算**

用法:
  python tools/ablation/aug_geo_audit.py --csv exp-std/csv/eval200_fixed.csv --n 200 \
      --out-txt /tmp/aug_audit.txt --out-png /tmp/aug_audit.png
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.ablation.build_aug_geo import _ink, _to_gray, aug_affine, aug_elastic, aug_grid, aug_morph  # noqa

FAMS = [
    ("morph r1-2", lambda ink, rng: aug_morph(ink, rng)),
    ("elastic a20", lambda ink, rng: aug_elastic(ink, rng, alpha=20.0, sigma=4.0)),
    ("elastic a10", lambda ink, rng: aug_elastic(ink, rng, alpha=10.0, sigma=4.0)),
    ("grid n4 d.1", lambda ink, rng: aug_grid(ink, rng, num_steps=4, distort_limit=0.1)),
    ("affine s3", lambda ink, rng: aug_affine(ink, rng, shear_deg=3.0)),
]


def stats(ink):
    """墨迹统计: 面积 / 连通分量数(≥8px) / 封闭洞数 / 笔宽代理"""
    b = ink > 0.5
    area = int(b.sum())
    lab, n = label(b)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        n_comp = int((sizes >= 8).sum())
    else:
        n_comp = 0
    bg = ~b
    lab2, n2 = label(bg)
    border = set(lab2[0, :]) | set(lab2[-1, :]) | set(lab2[:, 0]) | set(lab2[:, -1])
    n_hole = int(sum(1 for i in range(1, n2 + 1) if i not in border))
    w = float(distance_transform_edt(b).max()) if area else 0.0   # 最大内切半径
    wm = float(distance_transform_edt(b)[b].mean()) if area else 0.0
    return area, n_comp, n_hole, wm, w


def iou(a, b):
    A, B = a > 0.5, b > 0.5
    u = (A | B).sum()
    return float((A & B).sum()) / float(u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp-std/csv/eval200_fixed.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out-txt", default="/tmp/aug_audit.txt")
    ap.add_argument("--out-png", default="/tmp/aug_audit.png")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8"))
            if os.path.exists(os.path.join(a.root, r["image_path"]))][:a.n]
    print(f"  样本 {len(rows)} 张")

    acc = {f[0]: {k: [] for k in ("iou", "dcomp", "comp_chg", "dhole", "hole_chg",
                                  "area_r", "wm_r", "div")} for f in FAMS}
    tiles = []
    for ri, r in enumerate(rows):
        g = np.asarray(Image.open(os.path.join(a.root, r["image_path"])).convert("L"), dtype=np.uint8)
        base = _ink(g)
        a0, c0, h0, wm0, _ = stats(base)
        rng = np.random.default_rng(1234 + ri)
        row_tiles = [g]
        for name, fn in FAMS:
            v1, _ = fn(base, rng)
            v2, _ = fn(base, rng)
            a1, c1, h1, wm1, _ = stats(v1)
            d = acc[name]
            d["iou"].append(iou(base, v1))
            d["dcomp"].append(c1 - c0)
            d["comp_chg"].append(1.0 if c1 != c0 else 0.0)
            d["dhole"].append(h1 - h0)
            d["hole_chg"].append(1.0 if h1 != h0 else 0.0)
            d["area_r"].append(a1 / max(a0, 1))
            d["wm_r"].append(wm1 / max(wm0, 1e-6))
            d["div"].append(iou(v1, v2))
            if ri < 6:
                row_tiles.append(_to_gray(v1))
        if ri < 6:
            tiles.append(np.concatenate(row_tiles, axis=1))
        if (ri + 1) % 50 == 0:
            print(f"    {ri+1}/{len(rows)}")

    lines = []
    lines.append(f"增强审计  N={len(rows)}  (基准=原图; dcomp/dhole 为与原图之差)")
    lines.append(f"{'方案':<13}{'IoU↑':>7}{'Δ连通':>7}{'连通变%':>8}{'Δ洞':>7}{'洞变%':>7}"
                 f"{'面积比':>8}{'笔宽比':>8}{'两次增强IoU↓':>13}")
    for name, _ in FAMS:
        d = acc[name]
        lines.append(f"{name:<13}{np.mean(d['iou']):>7.3f}{np.mean(d['dcomp']):>7.2f}"
                     f"{100*np.mean(d['comp_chg']):>8.1f}{np.mean(d['dhole']):>7.2f}"
                     f"{100*np.mean(d['hole_chg']):>7.1f}{np.mean(d['area_r']):>8.3f}"
                     f"{np.mean(d['wm_r']):>8.3f}{np.mean(d['div']):>13.3f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(a.out_txt, "w", encoding="utf-8").write(txt + "\n")

    if tiles:
        grid = np.concatenate(tiles, axis=0)
        Image.fromarray(grid).save(a.out_png)
        print(f"  拼图 -> {a.out_png} ({grid.shape[1]}x{grid.shape[0]})  列: 原图 | " +
              " | ".join(n for n, _ in FAMS))


if __name__ == "__main__":
    sys.exit(main())
