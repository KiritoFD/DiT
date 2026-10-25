#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/compute_trainset_stroke_distribution.py —
在全量训练集 (train_top10_noaug.csv, 26,002 张真迹) 上测量并统计真实物理笔画粗细分布:
  - 杜绝测试集反向数据泄露 / Snooping
  - 多进程并行快速提取 (32 CPU workers)
  - 输出细粒度【书法家 × 书体】(23 个 slot) 统计量:
    Mean, Std, P10, P25, P50, P75, P90, Contrast (P90/P10), W_eq (等效笔宽)
  - 落盘为 JSON & CSV 供全生命周期使用
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis, skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "exp-std/csv/train_top10_noaug.csv"
OUT_JSON = ROOT / "assets/stroke_width_trainset_distribution.json"
OUT_CSV = ROOT / "assets/stroke_width_trainset_distribution.csv"


def process_single_image(args: tuple[int, str, str, str, str]) -> dict | None:
    """单个样本的笔画宽度特征提取。"""
    idx, rel_path, callig, script, char = args
    full_path = ROOT / rel_path
    if not full_path.exists():
        return None

    try:
        im = Image.open(full_path).convert("L")
        arr = np.array(im, dtype=np.float32) / 255.0

        # 背景为白(1.0)，墨迹为黑(0.0)
        ink_mask = (1.0 - arr) >= 0.5
        ink_sum = int(ink_mask.sum())
        if ink_sum < 25:
            return None

        dist_map = distance_transform_edt(ink_mask)
        skel = medial_axis(ink_mask)
        if not skel.any():
            skel = skeletonize(ink_mask)

        skel_len = int(skel.sum())
        if skel_len == 0:
            return None

        widths = 2.0 * dist_map[skel]
        widths = widths[widths >= 0.5]
        if len(widths) == 0:
            return None

        mean_w = float(np.mean(widths))
        std_w = float(np.std(widths))
        p10 = float(np.percentile(widths, 10))
        p25 = float(np.percentile(widths, 25))
        p50 = float(np.percentile(widths, 50))
        p75 = float(np.percentile(widths, 75))
        p90 = float(np.percentile(widths, 90))
        contrast = p90 / max(p10, 1e-4)
        w_eq = float(ink_sum) / max(float(skel_len), 1.0)

        return {
            "idx": idx,
            "calligrapher": callig,
            "script": script,
            "slot": f"{callig}_{script}",
            "char": char,
            "mean": mean_w,
            "std": std_w,
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "contrast": contrast,
            "w_eq": w_eq,
            "ink_area": ink_sum,
            "skel_len": skel_len,
        }
    except Exception:
        return None


def main():
    t0 = time.time()
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    print(f"Loaded {len(rows)} samples from training set: {CSV_PATH.name}")

    tasks = [
        (i, r["image_path"], r["calligrapher"], r["script"], r["character"])
        for i, r in enumerate(rows)
    ]

    num_workers = min(32, os.cpu_count() or 4)
    print(f"Extracting stroke properties using {num_workers} CPU workers...")

    results = []
    with Pool(num_workers) as pool:
        for res in pool.imap_unordered(process_single_image, tasks, chunksize=100):
            if res is not None:
                results.append(res)
                if len(results) % 5000 == 0:
                    print(f"  Processed {len(results)} / {len(tasks)} ({len(results)/len(tasks)*100:.1f}%) ...")

    print(f"Extraction completed: {len(results)} valid samples in {time.time() - t0:.1f}s.")

    # 分 slot 聚合统计
    slot_groups = defaultdict(list)
    for r in results:
        slot_groups[r["slot"]].append(r)

    dist_table = {}
    dist_rows = []

    for slot, items in sorted(slot_groups.items()):
        callig, script = slot.split("_")
        all_means = [x["mean"] for x in items]
        all_stds = [x["std"] for x in items]
        all_p10 = [x["p10"] for x in items]
        all_p25 = [x["p25"] for x in items]
        all_p50 = [x["p50"] for x in items]
        all_p75 = [x["p75"] for x in items]
        all_p90 = [x["p90"] for x in items]
        all_contrast = [x["contrast"] for x in items]
        all_weq = [x["w_eq"] for x in items]

        stat = {
            "slot": slot,
            "calligrapher": callig,
            "script": script,
            "sample_count": len(items),
            "ref_mean": float(np.mean(all_means)),
            "ref_std": float(np.mean(all_stds)),
            "ref_p10": float(np.mean(all_p10)),
            "ref_p25": float(np.mean(all_p25)),
            "ref_p50": float(np.mean(all_p50)),
            "ref_p75": float(np.mean(all_p75)),
            "ref_p90": float(np.mean(all_p90)),
            "ref_contrast": float(np.mean(all_contrast)),
            "ref_weq": float(np.mean(all_weq)),
        }
        dist_table[slot] = stat
        dist_rows.append(stat)

    # 全局总体统计
    all_means = [x["mean"] for x in results]
    all_stds = [x["std"] for x in results]
    all_p10 = [x["p10"] for x in results]
    all_p50 = [x["p50"] for x in results]
    all_p90 = [x["p90"] for x in results]
    dist_table["_GLOBAL_"] = {
        "slot": "_GLOBAL_",
        "calligrapher": "ALL",
        "script": "ALL",
        "sample_count": len(results),
        "ref_mean": float(np.mean(all_means)),
        "ref_std": float(np.mean(all_stds)),
        "ref_p10": float(np.mean(all_p10)),
        "ref_p25": float(np.mean([x["p25"] for x in results])),
        "ref_p50": float(np.mean(all_p50)),
        "ref_p75": float(np.mean([x["p75"] for x in results])),
        "ref_p90": float(np.mean(all_p90)),
        "ref_contrast": float(np.mean([x["contrast"] for x in results])),
        "ref_weq": float(np.mean([x["w_eq"] for x in results])),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dist_table, f, indent=2, ensure_ascii=False)

    df_out = pd.DataFrame(dist_rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved training set distribution table to:\n  {OUT_JSON}\n  {OUT_CSV}")

    print("\n" + "=" * 115)
    print("           【全量训练集 (26,002 真迹) 统计得出的细粒度真实粗细分布基线 (无泄露标准)】")
    print("=" * 115)
    print(df_out[[
        "calligrapher", "script", "sample_count",
        "ref_mean", "ref_std", "ref_p10", "ref_p50", "ref_p90", "ref_contrast"
    ]].to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    main()
