# -*- coding: utf-8 -*-
"""apply_clean_v2.py — 按"单笔画粗度"判据生成干净数据集 (复用 /tmp/scan_metrics.npz).

判据 (用户 2026-09-15 裁定: 单笔画粗度 40~80px; 这里取 40 作最严):
  K1 w_max > TH_W        # 最粗单笔宽度 (256x256 下 px)  -> 超粗笔画 / 黑块
  K2 ink   > 0.50        # 黑底白字 (背景是墨)
  K3 med   < 150         # 整体偏暗
  K4 blob  > 0.40        # 最大暗连通域占比 -> 外圈白 + 中间大黑块
动作: 脏图 -> data/_quarantine_v2/ ; 覆盖 assets/train_base_clean.csv /
      assets/train_base_sym_clean.csv ; shard 不动 (img_id 查表, csv 去掉即不读)
用法: python tools/apply_clean_v2.py --th 40 [--apply]
"""
import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(ROOT)

NOAUG = "assets/train_base_noaug.csv"
SYM = "assets/train_base_sym.csv"
OUT_NOAUG = "assets/train_base_clean.csv"
OUT_SYM = "assets/train_base_sym_clean.csv"
QUAR = "data/_quarantine_v2"
CACHE = "/tmp/scan_metrics.npz"
UID_T, UID_N = 7000000, 7100000

INK_HI, MED_LO, BLOB_HI = 0.50, 150.0, 0.25
# K5: 反色/黑底 —— 正常图的白底必然接触图像边缘; 反色图的亮部被黑包在中间,
#     边缘几乎全暗 (如反色的"一": 黑底 + 白色横条)。这是抓局部反色最准的判据。
EDGE_BRIGHT_LO = 0.50


def _montage(paths, out, cell=96, cols=8):
    sel = paths[:64]
    nrow = max((len(sel) + cols - 1) // cols, 1)
    cv = Image.new("RGB", (cols * cell, nrow * cell), (128, 0, 0))
    for i, p in enumerate(sel):
        try:
            a = np.asarray(Image.open(p).convert("L").resize((cell, cell)))
            cv.paste(Image.fromarray(a).convert("RGB"),
                     ((i % cols) * cell, (i // cols) * cell))
        except Exception:
            pass
    cv.save(out)
    print(f"  montage -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=float, default=40.0)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(NOAUG, encoding="utf-8")))
    paths = [r["image_path"] for r in rows]
    d = np.load(CACHE)
    w, ink, med, blob = d["w_max"], d["ink"], d["med"], d["blob"]
    eb = d["edge_bright"]
    assert len(w) == len(paths), f"cache mismatch {len(w)} vs {len(paths)}"

    # ★ 用户裁定 (2026-09-15): **唯一判据** —— 黑色部分(含外部包络)的最粗宽度 > 60px。
    #   其余判据 (ink/median/blob/edge_bright) 仅作为参考打印, 不参与剔除。
    dirty_mask = (w > a.th)
    dirty_idx = np.nonzero(dirty_mask)[0].tolist()
    clean_idx = np.nonzero(~dirty_mask)[0].tolist()
    print(f"[v2] th={a.th}px  base {len(paths)} -> clean {len(clean_idx)} "
          f"dirty {len(dirty_idx)}")
    n_k1 = int((w > a.th).sum())
    n_k2 = int((ink > INK_HI).sum())
    n_k3 = int((med < MED_LO).sum())
    n_k4 = int((blob > BLOB_HI).sum())
    n_k5 = int((eb < EDGE_BRIGHT_LO).sum())
    print(f"     命中数: K1(粗笔)={n_k1} K2(黑底)={n_k2} K3(暗)={n_k3} "
          f"K4(黑块)={n_k4} K5(反色)={n_k5}")

    def src_of(p):
        parts = p.split("/")
        return parts[2] if len(parts) > 2 else "misc"
    from collections import Counter
    c1 = Counter(src_of(paths[k]) for k in clean_idx)
    c0 = Counter(src_of(p) for p in paths)
    print("     按数据源 (clean/总):")
    for s in sorted(c0, key=lambda x: -c0[x]):
        print(f"       {s:22s} {c1.get(s,0):6d}/{c0[s]:6d}")

    os.makedirs("/tmp/cm3", exist_ok=True)
    _montage([paths[k] for k in clean_idx], "/tmp/cm3/clean.png")
    _montage([paths[k] for k in dirty_idx], "/tmp/cm3/dirty.png")
    _montage([paths[k] for k in dirty_idx if w[k] > a.th], "/tmp/cm3/k1_thick.png")
    _montage([paths[k] for k in dirty_idx if blob[k] > BLOB_HI], "/tmp/cm3/k4_blob.png")

    if not a.apply:
        print("[DRY-RUN] 加 --apply 执行。")
        return

    dirty_paths = {paths[k] for k in dirty_idx}
    os.makedirs(QUAR, exist_ok=True)
    moved = 0
    for k in dirty_idx:
        p = paths[k]
        if not os.path.exists(p):
            continue
        dst = os.path.join(QUAR, src_of(p))
        os.makedirs(dst, exist_ok=True)
        try:
            shutil.move(p, os.path.join(dst, os.path.basename(p)))
            moved += 1
        except Exception as e:
            print("  fail", p, e)
    # 增强兄弟
    path2idx = {r["image_path"]: k for k, r in enumerate(rows)}
    dirty_aug = set()
    for k in dirty_idx:
        kk = path2idx.get(paths[k])
        if kk is not None:
            dirty_aug.add(f"{UID_T + kk}.png")
            dirty_aug.add(f"{UID_N + kk}.png")
    aug_moved = 0
    os.makedirs(os.path.join(QUAR, "base_sym"), exist_ok=True)
    for nm in dirty_aug:
        q = f"data/imgs/base_sym/{nm}"
        if os.path.exists(q):
            shutil.move(q, os.path.join(QUAR, "base_sym", nm))
            aug_moved += 1
    print(f"[apply] moved base {moved}, aug {aug_moved}")

    fields = list(rows[0].keys())
    with open(OUT_NOAUG, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wtr.writeheader()
        for k in clean_idx:
            wtr.writerow(rows[k])

    sym = list(csv.DictReader(open(SYM, encoding="utf-8")))
    keep = []
    for r in sym:
        p = r["image_path"]
        if (r.get("aug") or "") == "":
            ok = p not in dirty_paths
        else:
            ok = os.path.basename(p) not in dirty_aug
        if ok:
            keep.append(r)
    sf = list(sym[0].keys())
    with open(OUT_SYM, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=sf, extrasaction="ignore")
        wtr.writeheader()
        for r in keep:
            wtr.writerow(r)
    n1 = sum(1 for _ in open(OUT_NOAUG, encoding="utf-8")) - 1
    n2 = sum(1 for _ in open(OUT_SYM, encoding="utf-8")) - 1
    print(f"[apply] {OUT_NOAUG}: {n1} 行")
    print(f"[apply] {OUT_SYM}: {n2} 行")


if __name__ == "__main__":
    main()
