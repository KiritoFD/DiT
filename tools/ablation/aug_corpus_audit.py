# -*- coding: utf-8 -*-
"""aug_corpus_audit.py — 审计**现有增广语料**是否破坏了字形拓扑

背景: build_aug_variants.py 的实现是「灰度 -> ink>0.5 二值化 -> 形态学 -> 高斯回软」。
      我新写的审计显示这套做法会切断抗锯齿细连接/抹掉浅洞（morph 43.9% 改连通、40.6% 改洞数）。
      若 v68/臂1/臂2 用的 77.8k 语料同样如此, 则"增广净负"可能是**实现坏**造成的假象。

做法: 抽 N 行增强样本, 用其 `image_path`(变体) 与 `src_image_path`(源图) 对比:
      IoU(墨迹) / Δ连通分量 / Δ封闭洞 / 面积比 / 笔宽比

用法: python tools/ablation/aug_corpus_audit.py --csv exp-std/csv/train_top10_aug_sym.csv --n 300
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


def load_gray(p):
    im = Image.open(p).convert("L")
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def ink_of(g):
    return 1.0 - g.astype(np.float32) / 255.0


def stats(ink):
    b = ink > 0.5
    area = int(b.sum())
    lab, n = label(b)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        n_comp = int((sizes >= 8).sum())
    else:
        n_comp = 0
    lab2, n2 = label(~b)
    border = set(lab2[0, :]) | set(lab2[-1, :]) | set(lab2[:, 0]) | set(lab2[:, -1])
    n_hole = int(sum(1 for i in range(1, n2 + 1) if i not in border))
    wm = float(distance_transform_edt(b)[b].mean()) if area else 0.0
    return area, n_comp, n_hole, wm


def iou(a, b):
    A, B = a > 0.5, b > 0.5
    u = (A | B).sum()
    return float((A & B).sum()) / float(u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp-std/csv/train_top10_aug_sym.csv")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"  语料 {a.csv}: {len(rows):,} 行")
    if rows and "aug" in rows[0]:
        from collections import Counter
        print(f"  aug 标签分布: {dict(Counter(r['aug'] for r in rows))}")
    # 配对: 旧脚本约定 `img_id = ID_BASE[tag] + 源img_id`
    #   （build_aug_variants.py: ID_BASE = {"sym": {"tp": 7_000_000, "tn": 7_100_000}}）
    #   语料没写 src_image_path, 故用这个确定性关系反推源图, 再用 img_id -> 行 查表。
    ID_BASE = {"tp": 7_000_000, "tn": 7_100_000, "tp1": 7_000_000, "tn1": 7_100_000}
    by_id = {}
    for r in rows:
        try:
            by_id[int(r["img_id"])] = r
        except Exception:
            pass

    def src_row(r):
        tag = (r.get("aug") or "").strip()
        if tag not in ID_BASE:
            return None
        try:
            sid = int(r["img_id"]) - ID_BASE[tag]
        except Exception:
            return None
        return by_id.get(sid)

    cand = []
    for r in rows:
        s = src_row(r)
        if s is None:
            continue
        if os.path.exists(os.path.join(a.root, r["image_path"])) and \
           os.path.exists(os.path.join(a.root, s["image_path"])):
            cand.append((r, s))
    print(f"  可审计增强行(用 img_id-ID_BASE 反推源图): {len(cand):,}")
    if not cand:
        print("  ✗ 没有可审计的行（检查 src_image_path / aug 列）")
        return 1
    step = max(1, len(cand) // a.n)
    sample = cand[::step][:a.n]
    print(f"  抽样 {len(sample)} 行")

    d_iou, d_comp, d_hole, r_area, r_w = [], [], [], [], []
    worst = []
    for r, s in sample:
        g1 = load_gray(os.path.join(a.root, r["image_path"]))
        g0 = load_gray(os.path.join(a.root, s["image_path"]))
        i1, i0 = ink_of(g1), ink_of(g0)
        a1, c1, h1, w1 = stats(i1)
        a0, c0, h0, w0 = stats(i0)
        d_iou.append(iou(i0, i1))
        d_comp.append(c1 - c0)
        d_hole.append(h1 - h0)
        r_area.append(a1 / max(a0, 1))
        r_w.append(w1 / max(w0, 1e-6))
        worst.append((abs(c1 - c0) + abs(h1 - h0), r["image_path"], s["image_path"],
                      c0, c1, h0, h1, round(iou(i0, i1), 3), r["aug"]))

    comp_chg = 100 * np.mean([1.0 if x != 0 else 0.0 for x in d_comp])
    hole_chg = 100 * np.mean([1.0 if x != 0 else 0.0 for x in d_hole])
    print()
    print("  ===== 现有增广语料审计（变体 vs 其源图） =====")
    print(f"  IoU(墨迹)     均值 = {np.mean(d_iou):.3f}   (1.0=完全没动)")
    print(f"  Δ连通分量     均值 = {np.mean(d_comp):+.3f}   改变比例 = {comp_chg:.1f}%   <-- 断笔指标")
    print(f"  Δ封闭洞       均值 = {np.mean(d_hole):+.3f}   改变比例 = {hole_chg:.1f}%   <-- 包围结构指标")
    print(f"  面积比        均值 = {np.mean(r_area):.3f}")
    print(f"  笔宽比        均值 = {np.mean(r_w):.3f}")
    print()
    print("  ===== 最严重的 8 例（|Δ连通|+|Δ洞| 排序） =====")
    worst.sort(reverse=True)
    for w in worst[:8]:
        print(f"    score={w[0]:.0f} comp {w[3]}->{w[4]} hole {w[5]}->{w[6]} IoU={w[7]} tag={w[8]} {w[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
