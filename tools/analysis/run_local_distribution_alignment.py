#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/run_local_distribution_alignment.py —
本地 GPU 驱动的强分布笔画粗细对齐引擎 (Spatially-Varying Quantile SDF):
  1. 读取全量训练集 (26,002 真迹) 统计得出的细粒度分布基线:
     assets/stroke_width_trainset_distribution.json
  2. 记录每个模型各样本的【原始粗细均值、方差、P10/P50/P90、提按比】
  3. 执行局部空间变异 SDF 笔画对齐:
     w_t(p) = mu_t + (sigma_t / sigma_g) * (w_g(p) - mu_g)
     delta(x) 空间扩散平滑 + 拓扑保护门禁
  4. 对齐历史 6 大 Checkpoints (Base 55k, DiC 50k, DiC 80k, Arm 2 50k, Exp A 50k, Exp A 70k)
  5. 本地 CUDA 计算 LPIPS(vgg) + SSIM + Wasserstein 距离 + 均值/方差误差收敛
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.stats import wasserstein_distance
from skimage.morphology import medial_axis, skeletonize
from skimage.metrics import structural_similarity as compute_ssim
import torch
import lpips

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "exp-std/csv/eval200_fixed.csv"
TRAIN_DIST_JSON = ROOT / "assets/stroke_width_trainset_distribution.json"

HISTORICAL_CKPTS = [
    ("Base 55k", ROOT / "artifacts/DiC_U100_review_20261020/images/originals/base_55k_ema"),
    ("DiC 50k", ROOT / "artifacts/DiC_U100_review_20261020/images/originals/dic_50k_cfg1.0"),
    ("DiC 80k", ROOT / "artifacts/DiC_U100_review_20261020/images/originals/dic_80k_cfg1.0"),
    ("Arm 2 50k", ROOT / "artifacts/DiC_U100_review_20261020/images/originals/scratch_writer_50k"),
    ("Exp A 50k", ROOT / "artifacts/DiC_U100_review_20261020/images/originals/decA_50k"),
    ("Exp A 70k", ROOT / "artifacts/DiC_U100_review_20261020/images/originals/decA_70k"),
]

GT_DIR = ROOT / "artifacts/DiC_U100_review_20261020/images/reference/gt"


def measure_stroke_properties(img_np: np.ndarray, threshold: float = 0.5) -> dict:
    """输入归一化灰度图 (1=白底, 0=黑字)，测量笔画几何分布与拓扑特征。"""
    ink_mask = (1.0 - img_np) >= threshold
    ink_sum = int(ink_mask.sum())
    if ink_sum < 25:
        return {
            "mean": 0.0, "std": 0.0,
            "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0,
            "contrast": 1.0, "w_eq": 0.0,
            "widths": np.array([0.0]), "ink_mask": ink_mask,
            "skel": np.zeros_like(ink_mask, dtype=bool),
            "dist_map": np.zeros_like(img_np),
            "num_fg": 0, "num_holes": 0, "ink_area": 0, "skel_len": 0
        }

    dist_map = distance_transform_edt(ink_mask)
    skel = medial_axis(ink_mask)
    if not skel.any():
        skel = skeletonize(ink_mask)

    if not skel.any():
        return {
            "mean": 0.0, "std": 0.0,
            "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0,
            "contrast": 1.0, "w_eq": 0.0,
            "widths": np.array([0.0]), "ink_mask": ink_mask,
            "skel": skel, "dist_map": dist_map,
            "num_fg": 1, "num_holes": 0, "ink_area": ink_sum, "skel_len": 0
        }

    widths = 2.0 * dist_map[skel]
    widths = widths[widths >= 0.5]
    if len(widths) == 0:
        widths = np.array([1.0])

    mean_w = float(np.mean(widths))
    std_w = float(np.std(widths))
    p10 = float(np.percentile(widths, 10))
    p25 = float(np.percentile(widths, 25))
    p50 = float(np.percentile(widths, 50))
    p75 = float(np.percentile(widths, 75))
    p90 = float(np.percentile(widths, 90))
    contrast = p90 / max(p10, 1e-4)

    skel_len = int(skel.sum())
    w_eq = float(ink_sum) / max(float(skel_len), 1.0)

    _, num_fg = label(ink_mask)
    _, num_bg = label(~ink_mask)
    num_holes = max(0, num_bg - 1)

    return {
        "mean": mean_w, "std": std_w,
        "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        "contrast": contrast, "w_eq": w_eq,
        "widths": widths, "ink_mask": ink_mask,
        "skel": skel, "dist_map": dist_map,
        "num_fg": num_fg, "num_holes": num_holes,
        "ink_area": ink_sum, "skel_len": skel_len
    }


def strong_distribution_sdf_align(
    img_np: np.ndarray,
    target_dist: dict,
    threshold: float = 0.5,
    max_shift: float = 2.5
) -> tuple[np.ndarray, dict, float]:
    """强分布对齐 (Spatially-Varying Quantile SDF):

    w_t(p) = mu_t + (sigma_t / sigma_g) * (w_g(p) - mu_g)
    精准匹配训练集真实均值与方差，兼顾提按对比度。
    """
    meas = measure_stroke_properties(img_np, threshold=threshold)
    orig_fg = meas["num_fg"]
    orig_holes = meas["num_holes"]
    skel = meas["skel"]
    widths = meas["widths"]

    if len(widths) < 10 or meas["mean"] < 1.0:
        return img_np, meas, 0.0

    mu_t = target_dist["ref_mean"]
    sigma_t = target_dist["ref_std"]
    mu_g = meas["mean"]
    sigma_g = max(meas["std"], 0.2)

    # 1. 分布变换缩放比
    scale = float(np.clip(sigma_t / sigma_g, 0.5, 2.0))

    # 2. 计算每个骨架点的理想粗细及位移 delta(p)
    skel_widths = 2.0 * meas["dist_map"][skel]
    target_widths = mu_t + scale * (skel_widths - mu_g)
    target_widths = np.clip(target_widths, 1.0, 45.0)

    delta_skel = (target_widths - skel_widths) / 2.0
    delta_skel = np.clip(delta_skel, -max_shift, max_shift)

    # 3. 空间扩散场构造
    # 将骨架点位移扩散至其法向临近域
    _, (nearest_r, nearest_c) = distance_transform_edt(~skel, return_indices=True)
    delta_grid = np.zeros_like(img_np, dtype=np.float32)
    delta_grid[skel] = delta_skel.astype(np.float32)
    delta_map = delta_grid[nearest_r, nearest_c]

    # 高斯平滑确保位移场平滑
    delta_map = gaussian_filter(delta_map, sigma=1.2)

    # 4. 有符号距离场 (SDF) 变形
    ink_mask = meas["ink_mask"]
    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    # 5. 拓扑保护与最佳增益搜索
    best_img = img_np
    best_meas = meas
    best_alpha = 0.0
    min_dist_err = abs(meas["mean"] - mu_t) + abs(meas["std"] - sigma_t)

    # 搜索增益梯度: 1.0(完全对齐) -> 0.75 -> 0.5 -> 0.25 -> 0.0
    for alpha in [1.0, 0.85, 0.70, 0.50, 0.30, 0.0]:
        phi_prime = phi + alpha * delta_map
        adjusted_ink = np.clip(phi_prime + 0.5, 0.0, 1.0)
        cand_img = (1.0 - adjusted_ink).astype(np.float32)

        cand_meas = measure_stroke_properties(cand_img, threshold=threshold)

        # 严格拓扑门禁:
        # a) 前景连通分支不得增加 (绝不断笔)
        if cand_meas["num_fg"] > orig_fg:
            continue
        # b) 孔洞不得严重闭合 (绝不糊字粘连)
        if cand_meas["num_holes"] < orig_holes - 1:
            continue

        dist_err = abs(cand_meas["mean"] - mu_t) + abs(cand_meas["std"] - sigma_t)
        if dist_err < min_dist_err:
            min_dist_err = dist_err
            best_img = cand_img
            best_meas = cand_meas
            best_alpha = alpha

    return best_img, best_meas, best_alpha


def main():
    t_start = time.time()
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    print(f"Loaded {len(rows)} samples from {CSV_PATH.name}")

    if not TRAIN_DIST_JSON.exists():
        print(f"ERROR: {TRAIN_DIST_JSON} not found!")
        return

    with open(TRAIN_DIST_JSON, "r", encoding="utf-8") as f:
        train_dist_table = json.load(f)
    print(f"Loaded training set distribution reference table: {len(train_dist_table)} slots.")

    # 初始化本地 GPU LPIPS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing LPIPS(vgg) on local {device} (RTX 4070 Laptop GPU)...")
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    # 预加载 GT 属性
    print("Measuring GT stroke distribution properties...")
    gt_stats = []
    for idx, r in enumerate(rows):
        gt_path = GT_DIR / f"g{idx}.png"
        if not gt_path.exists():
            gt_path = ROOT / r["image_path"]
        gt_im = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        meas = measure_stroke_properties(gt_im)
        meas["gt_im"] = gt_im
        gt_stats.append(meas)

    overall_summary = []

    # 循环对齐历史 6 大 Checkpoints
    for ckpt_name, ckpt_dir in HISTORICAL_CKPTS:
        if not ckpt_dir.exists():
            print(f"Skipping {ckpt_name}: not found at {ckpt_dir}")
            continue

        print(f"\n=======================================================")
        print(f"Processing Checkpoint: {ckpt_name} ({ckpt_dir.name})")
        print(f"=======================================================")

        out_aligned_dir = ROOT / f"assets/aligned_eval_results/{ckpt_name.replace(' ', '_')}_dist_aligned"
        out_aligned_dir.mkdir(parents=True, exist_ok=True)

        records = []

        for idx, r in enumerate(rows):
            callig = r["calligrapher"]
            script = r["script"]
            char = r["character"]
            slot = f"{callig}_{script}"

            target_dist = train_dist_table.get(slot, train_dist_table["_GLOBAL_"])

            gen_path = ckpt_dir / f"g{idx}.png"
            if not gen_path.exists():
                continue

            gen_im = np.array(Image.open(gen_path).convert("L"), dtype=np.float32) / 255.0
            gt_im = gt_stats[idx]["gt_im"]
            gt_meas = gt_stats[idx]

            # 强分布空间变异对齐
            aligned_im, aligned_meas, alpha = strong_distribution_sdf_align(gen_im, target_dist)
            orig_meas = measure_stroke_properties(gen_im)

            # 保存对齐后的生成图片
            out_png = out_aligned_dir / f"g{idx}.png"
            Image.fromarray((aligned_im * 255.0).astype(np.uint8)).save(out_png, optimize=True)

            # 评测 SSIM
            ssim_b = compute_ssim(gen_im, gt_im, data_range=1.0)
            ssim_a = compute_ssim(aligned_im, gt_im, data_range=1.0)

            # 评测 LPIPS (Local GPU)
            t_gt = torch.from_numpy(gt_im).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
            t_b = torch.from_numpy(gen_im).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
            t_a = torch.from_numpy(aligned_im).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0

            with torch.no_grad():
                lp_b = float(lpips_fn(t_b, t_gt).item())
                lp_a = float(lpips_fn(t_a, t_gt).item())

            wd_b = float(wasserstein_distance(orig_meas["widths"], gt_meas["widths"]))
            wd_a = float(wasserstein_distance(aligned_meas["widths"], gt_meas["widths"]))

            records.append({
                "idx": idx,
                "calligrapher": callig,
                "script": script,
                "slot": slot,
                "char": char,
                # 训练集真实目标
                "target_mean": target_dist["ref_mean"],
                "target_std": target_dist["ref_std"],
                "target_p50": target_dist["ref_p50"],
                # 原始生成分布
                "orig_mean": orig_meas["mean"],
                "orig_std": orig_meas["std"],
                "orig_p10": orig_meas["p10"],
                "orig_p50": orig_meas["p50"],
                "orig_p90": orig_meas["p90"],
                "orig_contrast": orig_meas["contrast"],
                # 对齐后生成分布
                "aligned_mean": aligned_meas["mean"],
                "aligned_std": aligned_meas["std"],
                "aligned_p10": aligned_meas["p10"],
                "aligned_p50": aligned_meas["p50"],
                "aligned_p90": aligned_meas["p90"],
                "aligned_contrast": aligned_meas["contrast"],
                "alpha": alpha,
                # 评测指标
                "ssim_before": ssim_b,
                "ssim_after": ssim_a,
                "lpips_before": lp_b,
                "lpips_after": lp_a,
                "wd_before": wd_b,
                "wd_after": wd_a,
            })

        df_m = pd.DataFrame(records)

        # 保存该 Checkpoint 的详细样本明细 CSV
        ckpt_csv = ROOT / f"assets/aligned_eval_results/{ckpt_name.replace(' ', '_')}_sample_audit.csv"
        df_m.to_csv(ckpt_csv, index=False, encoding="utf-8-sig")

        # 计算聚合统计
        mean_err_b = (abs(df_m["orig_mean"] - df_m["target_mean"]) / df_m["target_mean"]).mean() * 100
        mean_err_a = (abs(df_m["aligned_mean"] - df_m["target_mean"]) / df_m["target_mean"]).mean() * 100
        std_err_b = (abs(df_m["orig_std"] - df_m["target_std"]) / (df_m["target_std"] + 1e-4)).mean() * 100
        std_err_a = (abs(df_m["aligned_std"] - df_m["target_std"]) / (df_m["target_std"] + 1e-4)).mean() * 100

        p50_err_b = (abs(df_m["orig_p50"] - df_m["target_p50"]) / df_m["target_p50"]).mean() * 100
        p50_err_a = (abs(df_m["aligned_p50"] - df_m["target_p50"]) / df_m["target_p50"]).mean() * 100

        ssim_b_m = df_m["ssim_before"].mean()
        ssim_a_m = df_m["ssim_after"].mean()
        lp_b_m = df_m["lpips_before"].mean()
        lp_a_m = df_m["lpips_after"].mean()
        wd_b_m = df_m["wd_before"].mean()
        wd_a_m = df_m["wd_after"].mean()

        print(f"[{ckpt_name}] 强分布对齐成果:")
        print(f"  均值(Mean)误差: {mean_err_b:5.1f}% -> {mean_err_a:5.1f}% (收敛 {mean_err_b - mean_err_a:+.1f}%)")
        print(f"  方差(Std) 误差: {std_err_b:5.1f}% -> {std_err_a:5.1f}% (收敛 {std_err_b - std_err_a:+.1f}%)")
        print(f"  P50 典型笔宽误差: {p50_err_b:5.1f}% -> {p50_err_a:5.1f}% (收敛 {p50_err_b - p50_err_a:+.1f}%)")
        print(f"  SSIM 结构相似度:  {ssim_b_m:.4f} -> {ssim_a_m:.4f} ({ssim_a_m - ssim_b_m:+.4f})")
        print(f"  LPIPS 感知误差:   {lp_b_m:.4f} -> {lp_a_m:.4f} ({lp_a_m - lp_b_m:+.4f})")
        print(f"  Wasserstein 距离: {wd_b_m:.2f} px -> {wd_a_m:.2f} px ({wd_a_m - wd_b_m:+.2f} px)")

        overall_summary.append({
            "Checkpoint": ckpt_name,
            "原均值误差": f"{mean_err_b:.1f}%",
            "新均值误差": f"{mean_err_a:.1f}%",
            "原方差误差": f"{std_err_b:.1f}%",
            "新方差误差": f"{std_err_a:.1f}%",
            "原P50误差": f"{p50_err_b:.1f}%",
            "新P50误差": f"{p50_err_a:.1f}%",
            "SSIM变化": f"{ssim_b_m:.4f} -> {ssim_a_m:.4f} ({ssim_a_m - ssim_b_m:+.4f})",
            "LPIPS变化": f"{lp_b_m:.4f} -> {lp_a_m:.4f} ({lp_a_m - lp_b_m:+.4f})",
            "WD改善": f"{wd_b_m:.2f}px -> {wd_a_m:.2f}px ({wd_a_m - wd_b_m:+.2f}px)",
        })

    # ==============================================================
    # 最终汇总大表
    # ==============================================================
    print("\n" + "=" * 135)
    print("           【历史 6 大 Checkpoint 强分布对齐 (训练集基准 + 均值/方差/分位数) 终局对比大表】")
    print("=" * 135)
    df_summary = pd.DataFrame(overall_summary)
    print(df_summary.to_string(index=False))
    print("=" * 135)

    summary_csv = ROOT / "assets/aligned_eval_results/all_checkpoints_strong_distribution_summary.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved global summary to: {summary_csv.relative_to(ROOT)}")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
