#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/measure_calligrapher_stroke_width.py —
分书法家统计 GT 真迹与生成模型 (Exp A 70k, Arm 2 50k, Base 55k) 的笔画粗细指标:
  - Width-P10, Width-P50, Width-P90
  - Width-Contrast (P90 / P10)
  - W_eq (等效平均笔宽 = 墨迹面积 / 骨架总长度)
  - Width-Ratio (P50_gen / P50_gt)
  - Wasserstein 距离 (生成分布 vs GT分布)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import wasserstein_distance
from skimage.morphology import medial_axis, skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "exp-std/csv/eval200_fixed.csv"

EVAL_DIRS = {
    "Exp A 70k": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0070000/eval200fix",
    "Arm 2 50k": ROOT / "assets/results/unet_ablation_50k/arm2_writer_film_dec_b768/20261019-105029-Unet-flow/eval_samples_ctrl/step0050000/eval200fix",
    "Base 55k": ROOT / "assets/runs/v83A_55k_correct_eval_20261012/eval_samples_ctrl/step0055000/eval200fix",
}


def compute_stroke_widths(img_path: Path, threshold: float = 0.5) -> dict:
    """计算单张图像的笔画粗细分布及关键指标。

    约定：书法图背景为白色(1.0)，墨迹为黑色(0.0)；墨迹强度 = 1.0 - img_norm。
    """
    if not img_path.exists():
        return None

    im = Image.open(img_path).convert("L")
    arr = np.array(im, dtype=np.float32) / 255.0

    # 墨迹掩膜：黑字为前景 (ink >= threshold)
    ink_mask = (1.0 - arr) >= threshold

    if ink_mask.sum() < 20:  # 几乎空白
        return {
            "p10": 0.0, "p50": 0.0, "p90": 0.0,
            "contrast": 1.0, "w_eq": 0.0,
            "widths": np.array([0.0]), "ink_area": 0, "skel_len": 0
        }

    # 1. 距离变换 (计算墨迹内部每个点到背景的欧氏距离)
    dist_map = distance_transform_edt(ink_mask)

    # 2. 提取骨架中轴线
    skel = medial_axis(ink_mask)
    if not skel.any():
        skel = skeletonize(ink_mask)

    if not skel.any():
        return {
            "p10": 0.0, "p50": 0.0, "p90": 0.0,
            "contrast": 1.0, "w_eq": 0.0,
            "widths": np.array([0.0]), "ink_area": ink_mask.sum(), "skel_len": 0
        }

    # 3. 骨架点上的局部笔宽: w(p) ≈ 2 * D(p)
    widths = 2.0 * dist_map[skel]

    # 过滤异常毛刺 (例如 < 0.5px)
    widths = widths[widths >= 0.5]
    if len(widths) == 0:
        widths = np.array([1.0])

    p10 = float(np.percentile(widths, 10))
    p50 = float(np.percentile(widths, 50))
    p90 = float(np.percentile(widths, 90))
    contrast = p90 / max(p10, 1e-4)

    # 等效平均笔宽 W_eq = 墨迹面积 / 骨架总长度
    ink_area = float(ink_mask.sum())
    skel_len = float(skel.sum())
    w_eq = ink_area / max(skel_len, 1.0)

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "contrast": contrast,
        "w_eq": w_eq,
        "widths": widths,
        "ink_area": ink_area,
        "skel_len": skel_len,
    }


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    n_total = len(rows)
    print(f"Loaded {n_total} rows from {CSV_PATH.name}")

    # 检查各模型存在性
    valid_models = {}
    for m_name, m_dir in EVAL_DIRS.items():
        if m_dir.exists():
            valid_models[m_name] = m_dir
            print(f"Found model dir [{m_name}]: {m_dir.relative_to(ROOT)}")
        else:
            print(f"Skipping missing model [{m_name}]: {m_dir}")

    # 1. 逐图计算 GT 与各模型的笔宽
    print("\nComputing stroke widths for GT and models...")
    records = []

    for idx, r in enumerate(rows):
        callig = r["calligrapher"]
        script = r["script"]
        char = r["character"]
        gt_path = ROOT / r["image_path"]

        gt_stat = compute_stroke_widths(gt_path)
        if gt_stat is None:
            continue

        rec = {
            "idx": idx,
            "calligrapher": callig,
            "script": script,
            "character": char,
            "gt_p10": gt_stat["p10"],
            "gt_p50": gt_stat["p50"],
            "gt_p90": gt_stat["p90"],
            "gt_contrast": gt_stat["contrast"],
            "gt_weq": gt_stat["w_eq"],
            "gt_widths": gt_stat["widths"],
        }

        for m_name, m_dir in valid_models.items():
            gen_path = m_dir / f"g{idx}.png"
            gen_stat = compute_stroke_widths(gen_path)
            if gen_stat is not None:
                rec[f"{m_name}_p10"] = gen_stat["p10"]
                rec[f"{m_name}_p50"] = gen_stat["p50"]
                rec[f"{m_name}_p90"] = gen_stat["p90"]
                rec[f"{m_name}_contrast"] = gen_stat["contrast"]
                rec[f"{m_name}_weq"] = gen_stat["w_eq"]
                rec[f"{m_name}_ratio"] = gen_stat["p50"] / max(gt_stat["p50"], 1e-4)
                rec[f"{m_name}_wd"] = float(wasserstein_distance(gen_stat["widths"], gt_stat["widths"]))

        records.append(rec)

    df_res = pd.DataFrame(records)
    print(f"Successfully processed {len(df_res)} valid samples.")

    # 2. 分书法家聚合统计
    print("\n" + "=" * 95)
    print("                      【各书法家 GT 真迹笔画粗细物理特征基线】")
    print("=" * 95)
    summary_gt = []
    for callig, g in df_res.groupby("calligrapher"):
        summary_gt.append({
            "书法家": callig,
            "样本数": len(g),
            "P10(细笔)": f"{g['gt_p10'].mean():.2f} px",
            "P50(典型笔宽)": f"{g['gt_p50'].mean():.2f} px",
            "P90(重墨粗笔)": f"{g['gt_p90'].mean():.2f} px",
            "提按比(P90/P10)": f"{g['gt_contrast'].mean():.2f}",
            "等效笔宽(Weq)": f"{g['gt_weq'].mean():.2f} px",
        })
    df_gt_tab = pd.DataFrame(summary_gt).sort_values("P50(典型笔宽)", ascending=False)
    print(df_gt_tab.to_string(index=False))

    # 全局 GT 基准
    print("-" * 95)
    print(f"【全体平均 GT】: P10={df_res['gt_p10'].mean():.2f} px | P50={df_res['gt_p50'].mean():.2f} px | "
          f"P90={df_res['gt_p90'].mean():.2f} px | Contrast={df_res['gt_contrast'].mean():.2f} | Weq={df_res['gt_weq'].mean():.2f} px")
    print("=" * 95)

    # 3. 各模型 vs GT 对比表格
    for m_name in valid_models.keys():
        print("\n" + "=" * 115)
        print(f"                      【模型对比: {m_name} vs GT 真迹笔宽与提按分布】")
        print("=" * 115)
        m_rows = []
        for callig, g in df_res.groupby("calligrapher"):
            gt_p50 = g["gt_p50"].mean()
            m_p50 = g[f"{m_name}_p50"].mean()
            m_p10 = g[f"{m_name}_p10"].mean()
            m_p90 = g[f"{m_name}_p90"].mean()
            gt_con = g["gt_contrast"].mean()
            m_con = g[f"{m_name}_contrast"].mean()
            ratio = g[f"{m_name}_ratio"].mean()
            wd = g[f"{m_name}_wd"].mean()

            diff_p50_pct = ((m_p50 - gt_p50) / gt_p50) * 100

            m_rows.append({
                "书法家": callig,
                "GT P50": f"{gt_p50:.2f}px",
                f"Gen P50": f"{m_p50:.2f}px",
                "P50误差": f"{diff_p50_pct:+.1f}%",
                "Gen P10": f"{m_p10:.2f}px",
                "Gen P90": f"{m_p90:.2f}px",
                "GT提按比": f"{gt_con:.2f}",
                "Gen提按比": f"{m_con:.2f}",
                "Width-Ratio": f"{ratio:.3f}",
                "Wasserstein距": f"{wd:.2f}px",
            })
        df_m_tab = pd.DataFrame(m_rows)
        print(df_m_tab.to_string(index=False))

        # 整体
        tot_gt_p50 = df_res["gt_p50"].mean()
        tot_m_p50 = df_res[f"{m_name}_p50"].mean()
        tot_diff = ((tot_m_p50 - tot_gt_p50) / tot_gt_p50) * 100
        print("-" * 115)
        print(f"【全体汇总】 GT P50={tot_gt_p50:.2f}px | Gen P50={tot_m_p50:.2f}px ({tot_diff:+.1f}%) | "
              f"GT提按比={df_res['gt_contrast'].mean():.2f} vs Gen提按比={df_res[f'{m_name}_contrast'].mean():.2f} | "
              f"平均 Wasserstein 分布距离: {df_res[f'{m_name}_wd'].mean():.2f}px")
        print("=" * 115)

    # 4. 保存详细明细 CSV
    out_csv = ROOT / "scratch/stroke_width_per_sample_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in df_res.columns if c not in ("gt_widths")]
    df_res[export_cols].to_csv(out_csv, index=False)
    print(f"\nDetailed per-sample measurements saved to: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
