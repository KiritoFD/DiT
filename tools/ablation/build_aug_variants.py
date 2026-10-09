#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_aug_variants.py — 通用「笔画形态」数据增强构建器（消融实验用）。

为什么只做形态学增强
--------------------
`tools/dino_expand_top10_aug.py` 的判据：只要增强**不改变字形/位置/书家/书体**，
DINOv2 的 patch-token 基本不变，于是 REPA 缓存可以合法地「增强行复用源图特征」，
零额外前向、且保住配方单变量。本脚本只做 **二值形态学（笔画粗细）** 增强。
⚠ 几何类增强（仿射/弹性/裁剪/旋转）会改变 DINO 特征，**不能**复用 REPA 缓存，
   本脚本不提供；如将来确需，必须另外用 DINOv2 重算增强图的特征。

与历史脚本的关系
----------------
* `sym` 策略逐字节复刻 `tools/build_top10_sym_aug.py`（v4：tp/tn，同一原图 ± 对同 p，
  p = 1+((idx*2654435761)%2) ∈ {1,2}，8-邻域，保护逻辑一致）→ 即 v68 的 3x 数据。
* `sym5x` 采用 48 上 `train_top10_aug_sym5x.csv` 观测到的约定
  （tp1=7.0M, tn1=7.1M, tp2=7.2M, tn2=7.3M）→ 即 v69 的 5x 数据。
  （历史 5x 构建脚本已不在仓库，重生成可能与 v69 的图集略有出入，但 id/tag 约定对齐。）

产物
----
1. 增强图像目录（PNG, 256x256, L 模式, 白纸黑墨）
2. 全景 CSV（原图 + 各变体，`aug` 列标注 tag，`img_id` 分段不冲突）
3. manifest JSON（记录 tag -> img_id 基址，供 REPA 扩展 / 编码 / 配置生成复用）

用法
----
  python tools/ablation/build_aug_variants.py \
      --strategy sym5x --root . \
      --src-csv exp-std/csv/train.csv \
      --out-csv exp-std/csv/train_top10_aug_sym5x.csv \
      --imgs-dir data/top10_style23/imgs_aug_sym5x \
      --workers 32
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

# tag -> (op, p)   op ∈ {dilate, erode}; p = 整数 或 "auto"(v4 的确定性哈希)
STRATEGIES = {
    "sym": [("tp", "dilate", "auto"), ("tn", "erode", "auto")],
    "sym5x": [("tp1", "dilate", 1), ("tn1", "erode", 1),
              ("tp2", "dilate", 2), ("tn2", "erode", 2)],
    "thick": [("tp1", "dilate", 1), ("tp2", "dilate", 2)],
    "thin": [("tn1", "erode", 1), ("tn2", "erode", 2)],
    "symwide": [("tp1", "dilate", 1), ("tn1", "erode", 1),
                ("tp2", "dilate", 2), ("tn2", "erode", 2),
                ("tp3", "dilate", 3), ("tn3", "erode", 3)],
    "sym4": [("tp4", "dilate", 1), ("tn4", "erode", 1)],   # 4-邻域十字结构元
}
# tag -> img_id 加成基址。每段留 100k 空间（原图 id 最大 38,583）。
ID_BASE = {
    "sym": {"tp": 7_000_000, "tn": 7_100_000},
    "sym5x": {"tp1": 7_000_000, "tn1": 7_100_000, "tp2": 7_200_000, "tn2": 7_300_000},
    "thick": {"tp1": 8_000_000, "tp2": 8_100_000},
    "thin": {"tn1": 8_200_000, "tn2": 8_300_000},
    "symwide": {"tp1": 8_400_000, "tn1": 8_500_000, "tp2": 8_600_000, "tn2": 8_700_000,
                "tp3": 8_800_000, "tn3": 8_900_000},
    "sym4": {"tp4": 9_000_000, "tn4": 9_100_000},
}
STRUCT = {"8": generate_binary_structure(2, 2), "4": generate_binary_structure(2, 1)}

# 保护阈值（与 v4 完全一致）
THIN_MIN_FRAC, THIN_MIN_PX = 0.15, 20
THICK_MAX_MULT = 3.0

_G = {}


def _init(root, imgs_dir, jobs, force):
    _G["root"] = root
    _G["imgs_dir"] = imgs_dir
    _G["jobs"] = jobs          # [(tag, op, p)]
    _G["force"] = force
    _G["st"] = STRUCT[jobs_meta["st"]]


jobs_meta = {"st": "8"}


def _auto_p(idx):
    return 1 + ((idx * 2654435761) % 2)


def _morph(ink, op, p, st):
    m = binary_dilation(ink, structure=st, iterations=p) if op == "dilate" \
        else binary_erosion(ink, structure=st, iterations=p)
    return m


def process_row(task):
    idx, row = task
    root = _G["root"]
    imgs_dir = _G["imgs_dir"]
    st = _G["st"]
    force = _G["force"]

    try:
        base_id = int(row["img_id"])
    except Exception:
        return idx, {}

    src_abs = os.path.join(root, row["image_path"])
    out = {}
    try:
        im = Image.open(src_abs).convert("L")
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.BICUBIC)
        g = np.asarray(im)
    except Exception:
        return idx, out

    ink = g < 128
    orig_area = int(ink.sum())
    if orig_area < THIN_MIN_PX:
        return idx, out

    p_auto = _auto_p(idx)
    for tag, op, pspec in _G["jobs"]:
        fname = f"{base_id:06d}_{tag}.png"
        abs_p = os.path.join(imgs_dir, fname)
        rel_p = os.path.relpath(abs_p, root).replace(os.sep, "/")
        if os.path.exists(abs_p) and not force:
            out[tag] = rel_p
            continue

        p0 = p_auto if pspec == "auto" else int(pspec)
        p = p0
        m = None
        ok = False
        while p > 0:
            m = _morph(ink, op, p, st)
            a = int(m.sum())
            if op == "dilate":
                if a <= THICK_MAX_MULT * orig_area:
                    ok = True
                    break
            else:
                if a >= THIN_MIN_FRAC * orig_area and a >= THIN_MIN_PX:
                    ok = True
                    break
            p -= 1
        if ok and m is not None:
            arr = np.where(m, 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(abs_p, "PNG", compress_level=1)
            out[tag] = rel_p
    return idx, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    ap.add_argument("--root", default=".", help="仓库根（image_path 相对它解析）")
    ap.add_argument("--src-csv", default="exp-std/csv/train.csv")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--imgs-dir", default=None)
    ap.add_argument("--struct", default="8", choices=["8", "4"])
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 行（冒烟测试）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的变体图")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    jobs = STRATEGIES[a.strategy]
    if a.imgs_dir is None:
        a.imgs_dir = f"data/top10_style23/imgs_aug_{a.strategy}"
    if a.out_csv is None:
        a.out_csv = f"exp-std/csv/train_top10_aug_{a.strategy}.csv"
    imgs_dir = os.path.join(root, a.imgs_dir)
    os.makedirs(imgs_dir, exist_ok=True)
    jobs_meta["st"] = a.struct

    print("=" * 78)
    print(f"[aug] strategy={a.strategy} struct={a.struct}")
    print(f"[aug] src={os.path.join(root, a.src_csv)}")
    print(f"[aug] imgs_dir={imgs_dir}")
    print(f"[aug] out_csv={os.path.join(root, a.out_csv)}")
    print(f"[aug] jobs={jobs}  id_base={ID_BASE[a.strategy]}")
    print("=" * 78, flush=True)

    rows = list(csv.DictReader(open(os.path.join(root, a.src_csv), encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    n_src = len(rows)
    print(f"[aug] 源行数 {n_src:,}")

    if "img_id" not in (rows[0] if rows else {}):
        print("[aug] FATAL: 源 CSV 缺 img_id 列，拒绝用文件名正则兜底（会静默错位）")
        return 2

    tasks = list(enumerate(rows))
    t0 = time.time()
    res = {}
    with mp.Pool(min(a.workers, max(1, os.cpu_count() or 1)),
                 initializer=_init, initargs=(root, imgs_dir, jobs, a.force)) as pool:
        for idx, out in pool.imap_unordered(process_row, tasks, chunksize=128):
            res[idx] = out
            if (idx + 1) % 5000 == 0 or (idx + 1) == n_src:
                print(f"[aug]   {idx + 1:,}/{n_src:,} "
                      f"({(idx + 1) / max(1e-6, time.time() - t0):.1f} 图/s)", flush=True)

    counts = {tag: sum(1 for v in res.values() if v.get(tag)) for tag, _, _ in jobs}
    print("=" * 78)
    print(f"[aug] 变体生成完毕，耗时 {time.time() - t0:.1f}s")
    for tag, c in counts.items():
        print(f"    {tag:5s}: {c:,} ({c / max(1, n_src) * 100:.2f}%)")
    n_out = n_src + sum(counts.values())
    print(f"[aug] 总行数 {n_out:,} (原图 {n_src:,} + 变体 {sum(counts.values()):,})")

    fields = list(rows[0].keys())
    if "aug" not in fields:
        fields.append("aug")
    out_rows = []
    for idx, r in enumerate(rows):
        r0 = dict(r)
        r0["aug"] = ""
        out_rows.append(r0)
        for tag, _, _ in jobs:
            rel = res.get(idx, {}).get(tag)
            if rel:
                ra = dict(r)
                ra["image_path"] = rel
                ra["aug"] = tag
                ra["img_id"] = ID_BASE[a.strategy][tag] + int(r["img_id"])
                out_rows.append(ra)
    out_csv_abs = os.path.join(root, a.out_csv)
    os.makedirs(os.path.dirname(out_csv_abs), exist_ok=True)
    with open(out_csv_abs, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[aug] CSV 写入 {out_csv_abs} ({len(out_rows):,} 行)")

    ids = [int(r["img_id"]) for r in out_rows]
    assert len(set(ids)) == len(ids), "img_id 有重复！"
    manifest = {
        "strategy": a.strategy, "struct": a.struct,
        "src_csv": a.src_csv, "out_csv": a.out_csv, "imgs_dir": a.imgs_dir,
        "id_base": ID_BASE[a.strategy],
        "tags": [t for t, _, _ in jobs],
        "n_src": n_src, "n_rows": len(out_rows), "counts": counts,
        "geometry_preserving": True,
        "repa_note": "形态学增强，DINO patch-token 基本不变 -> REPA 可复用源图特征",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    man_path = out_csv_abs + ".manifest.json"
    json.dump(manifest, open(man_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[aug] manifest 写入 {man_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
