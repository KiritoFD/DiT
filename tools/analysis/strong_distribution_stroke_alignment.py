#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/strong_distribution_stroke_alignment.py —
强分布笔画粗细对齐引擎 (Spatially-Varying Quantile SDF):
  1. 细粒度【书法家 × 书体】完整分布特征库持久化 (Mean, Std, P10, P50, P90, Contrast, Weq)
  2. 空间变异 SDF 笔画对齐:
     - 测量每个骨架点局部宽度 w(p)
     - 分布矩/分位数映射: w_target(p) = mu_t + (sigma_t / sigma_g) * (w(p) - mu_g)
     - 空间扩散构建光滑位移场 delta(x)
     - 带有连通域与孔洞拓扑保护门禁
  3. 全量处理 7 个历史 Checkpoint (Base 55k, Arm 2 50k, Exp A 50k/55k/60k/65k/70k)
  4. 完整输出对齐前后全套指标 (均值误差, 方差误差, SSIM, LPIPS, Wasserstein 距)
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
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.stats import wasserstein_distance
from skimage.morphology import medial_axis, skeletonize
from skimage.metrics import structural_similarity as compute_ssim
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "exp-std/csv/eval200_fixed.csv"
DIST_TABLE_JSON = ROOT / "assets/stroke_width_distribution_table.json"
DIST_TABLE_CSV = ROOT / "assets/stroke_width_distribution_table.csv"

TARGET_CHECKPOINTS = [
    ("Base 55k", ROOT / "assets/runs/v83A_55k_correct_eval_20261012/eval_samples_ctrl/step0055000/eval200fix"),
    ("Arm 2 50k", ROOT / "assets/results/unet_ablation_50k/arm2_writer_film_dec_b768/20261019-105029-Unet-flow/eval_samples_ctrl/step0050000/eval200fix"),
    ("Exp A 50k", ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_50k/20261019-191851-Unet-flow/eval_samples_ctrl/step0050000/eval200fix"),
    ("Exp A 55k", ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0055000/eval200fix"),
    ("Exp A 60k", ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0060000/eval200fix"),
    ("Exp A 65k", ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0065000/eval200fix"),
    ("Exp A 70k", ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0070000/eval200fix"),
]


def measure_stroke_distribution(img_np: np.ndarray, threshold: float = 0.5) -> dict:
    """输入灰度图 (1=白底, 0=黑字)，测量笔画几何分布与拓扑特征。"""
    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 20:
        return {
            "mean": 0.0, "std": 0.0,
            "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0,
            "contrast": 1.0, "w_eq": 0.0,
            "widths": np.array([0.0]), "ink_mask": ink_mask,
            "skel": np.zeros_like(ink_mask, dtype=bool),
            "dist_map": np.zeros_like(img_np),
            "num_fg": 0, "num_holes": 0
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
            "num_fg": 1, "num_holes": 0
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

    ink_area = float(ink_mask.sum())
    skel_len = float(skel.sum())
    w_eq = ink_area / max(skel_len, 1.0)

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
    }


def strong_distribution_sdf_align(
    img_np: np.ndarray,
    target_dist: dict,
    threshold: float = 0.5,
    max_shift: float = 2.0
) -> tuple[np.ndarray, dict]:
    """利用空间变异 SDF (Spatially Varying SDF) 实现强分布对齐 (均值 + 方差 + 分位数)。"""
    meas = measure_stroke_distribution(img_np, threshold=threshold)
    orig_fg = meas["num_fg"]
    orig_holes = meas["num_holes"]
    skel = meas["skel"]
    widths = meas["widths"]

    if len(widths) < 10 or meas["mean"] < 1.0:
        return img_np, meas

    mu_t = target_dist["ref_mean"]
    sigma_t = target_dist["ref_std"]
    mu_g = meas["mean"]
    sigma_g = max(meas["std"], 1e-3)

    # 1. 目标分布映射函数: 让生成笔宽分布精准映射到目标 (均值 mu_t, 方差 sigma_t)
    # 对极端异常值做 3-sigma 软截断保护
    scale = np.clip(sigma_t / sigma_g, 0.4, 2.5)
    
    # 骨架点坐标
    skel_coords = np.argwhere(skel)  # (N, 2)
    skel_widths = 2.0 * meas["dist_map"][skel]
    
    # 每个骨架点的理想目标笔宽与位移量
    target_widths = mu_t + scale * (skel_widths - mu_g)
    target_widths = np.clip(target_widths, 1.0, 45.0)
    delta_skel = (target_widths - skel_widths) / 2.0
    delta_skel = np.clip(delta_skel, -max_shift, max_shift)

    # 2. 空间扩散: 利用最近骨架点变换 (Nearest Skeleton Point) 将位移扩散到全图
    # 计算全图每个像素到最近骨架点的索引
    _, (nearest_r, nearest_c) = distance_transform_edt(~skel, return_indices=True)
    
    # 构建骨架点到位移的查找表
    delta_grid = np.zeros_like(img_np, dtype=np.float32)
    delta_grid[skel] = delta_skel.astype(np.float32)
    
    # 广播到全图
    delta_map = delta_grid[nearest_r, nearest_c]
    
    # 空间高斯平滑，保证法向偏移场的光滑连续性
    delta_map = gaussian_filter(delta_map, sigma=1.5)

    # 3. 有符号距离场变形与亚像素抗锯齿渲染
    ink_mask = meas["ink_mask"]
    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    # 搜索最优增益 alpha in [0.5, 1.0, 1.25] 并实施拓扑保护
    best_img = img_np
    best_meas = meas
    min_dist_err = abs(meas["mean"] - mu_t) + abs(meas["std"] - sigma_t)

    for alpha in [1.0, 0.75, 0.5, 1.25, 0.0]:
        phi_prime = phi + alpha * delta_map
        adjusted_ink = np.clip(phi_prime + 0.5, 0.0, 1.0)
        cand_img = (1.0 - adjusted_ink).astype(np.float32)

        cand_meas = measure_stroke_distribution(cand_img, threshold=threshold)

        # 拓扑门禁: 不能断笔，孔洞不能被过度闭合
        if cand_meas["num_fg"] > orig_fg:  # 发生断笔
            continue
        if cand_meas["num_holes"] < orig_holes - 1:  # 孔洞被粘连糊死
            continue

        dist_err = abs(cand_meas["mean"] - mu_t) + abs(cand_meas["std"] - sigma_t)
        if dist_err < min_dist_err:
            min_dist_err = dist_err
            best_img = cand_img
            best_meas = cand_meas

    return best_img, best_meas


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    print(f"Loaded {len(rows)} samples from {CSV_PATH.name}")

    # ==============================================================
    # 步骤 1: 构建并持久化【书法家 × 书体】完整分布特征库
    # ==============================================================
    print("\n[Step 1] Constructing Full Distribution Reference Table across Calligrapher * Script...")
    slot_gt_measurements = defaultdict(list)
    per_sample_gt = []

    for idx, r in enumerate(rows):
        callig = r["calligrapher"]
        script = r["script"]
        gt_path = ROOT / r["image_path"]

        im_np = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        meas = measure_stroke_distribution(im_np)
        meas["idx"] = idx
        meas["calligrapher"] = callig
        meas["script"] = script
        meas["character"] = r["character"]
        meas["gt_path"] = str(gt_path.relative_to(ROOT))
        per_sample_gt.append(meas)

        slot = f"{callig}_{script}"
        slot_gt_measurements[slot].append(meas)

    dist_table = {}
    dist_table_rows = []
    for slot, m_list in sorted(slot_gt_measurements.items()):
        callig, script = slot.split("_")
        all_means = [m["mean"] for m in m_list]
        all_stds = [m["std"] for m in m_list]
        all_p10 = [m["p10"] for m in m_list]
        all_p25 = [m["p25"] for m in m_list]
        all_p50 = [m["p50"] for m in m_list]
        all_p75 = [m["p75"] for m in m_list]
        all_p90 = [m["p90"] for m in m_list]
        all_contrast = [m["contrast"] for m in m_list]
        all_weq = [m["w_eq"] for m in m_list]

        slot_dist = {
            "slot": slot,
            "calligrapher": callig,
            "script": script,
            "sample_count": len(m_list),
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
        dist_table[slot] = slot_dist
        dist_table_rows.append(slot_dist)

    DIST_TABLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DIST_TABLE_JSON, "w", encoding="utf-8") as f:
        json.dump(dist_table, f, indent=2, ensure_ascii=False)

    df_dist = pd.DataFrame(dist_table_rows)
    df_dist.to_csv(DIST_TABLE_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved distribution reference table to: {DIST_TABLE_JSON.relative_to(ROOT)} and {DIST_TABLE_CSV.relative_to(ROOT)}")

    print("\n" + "=" * 115)
    print("                     【细粒度书法家 × 书体 GT 笔画粗细分布特征库 (均值/方差/分位数)】")
    print("=" * 115)
    print(df_dist[[
        "calligrapher", "script", "sample_count",
        "ref_mean", "ref_std", "ref_p10", "ref_p50", "ref_p90", "ref_contrast"
    ]].to_string(index=False))
    print("=" * 115)

    # ==============================================================
    # 步骤 2: 对 7 个历史 Checkpoint 执行强分布对齐并评测
    # ==============================================================
    print("\n[Step 2] Executing Strong Distribution Alignment on 7 Historical Checkpoints...")

    from src.eval.decoder import _get_lpips
    lpips_fn = _get_lpips()
    ldev = next(lpips_fn.parameters()).device if lpips_fn else "cpu"
    print(f"LPIPS engine ready on device: {ldev}")

    ckpt_results = []

    for name, c_dir in TARGET_CHECKPOINTS:
        if not c_dir.exists():
            print(f"Skipping {name}: dir not found ({c_dir})")
            continue

        print(f"\n>>> Processing Checkpoint: {name} ...")
        aligned_out_dir = c_dir.parent / f"{c_dir.name}_strong_dist_aligned"
        aligned_out_dir.mkdir(parents=True, exist_ok=True)

        sample_evals = []

        for idx, r in enumerate(rows):
            callig = r["calligrapher"]
            script = r["script"]
            char = r["character"]
            slot = f"{callig}_{script}"
            target_dist = dist_table[slot]

            gen_path = c_dir / f"g{idx}.png"
            if not gen_path.exists():
                continue

            gt_meas = per_sample_gt[idx]
            gt_im = np.array(Image.open(ROOT / r["image_path"]).convert("L"), dtype=np.float32) / 255.0
            gen_im = np.array(Image.open(gen_path).convert("L"), dtype=np.float32) / 255.0

            orig_meas = measure_stroke_distribution(gen_im)

            # 强分布对齐 (空间变异 Quantile SDF + 拓扑门禁)
            aligned_im, aligned_meas = strong_distribution_sdf_align(gen_im, target_dist)

            # 落盘保存对齐后的图像
            out_png = aligned_out_dir / f"g{idx}.png"
            Image.fromarray((aligned_im * 255.0).astype(np.uint8)).save(out_png, optimize=True)

            # 计算 SSIM
            ssim_b = compute_ssim(gen_im, gt_im, data_range=1.0)
            ssim_a = compute_ssim(aligned_im, gt_im, data_range=1.0)

            # 计算 LPIPS
            if lpips_fn is not None:
                t_gt = torch.from_numpy(gt_im).unsqueeze(0).repeat(1, 3, 1, 1).to(ldev) * 2.0 - 1.0
                t_b = torch.from_numpy(gen_im).unsqueeze(0).repeat(1, 3, 1, 1).to(ldev) * 2.0 - 1.0
                t_a = torch.from_numpy(aligned_im).unsqueeze(0).repeat(1, 3, 1, 1).to(ldev) * 2.0 - 1.0
                with torch.no_grad():
                    lpips_b = float(lpips_fn(t_b, t_gt).item())
                    lpips_a = float(lpips_fn(t_a, t_gt).item())
            else:
                lpips_b, lpips_a = 0.0, 0.0

            wd_b = float(wasserstein_distance(orig_meas["widths"], gt_meas["widths"]))
            wd_a = float(wasserstein_distance(aligned_meas["widths"], gt_meas["widths"]))

            sample_evals.append({
                "idx": idx,
                "calligrapher": callig,
                "script": script,
                "slot": slot,
                "char": char,
                "target_mean": target_dist["ref_mean"],
                "target_std": target_dist["ref_std"],
                "orig_mean": orig_meas["mean"],
                "aligned_mean": aligned_meas["mean"],
                "orig_std": orig_meas["std"],
                "aligned_std": aligned_meas["std"],
                "orig_p50": orig_meas["p50"],
                "aligned_p50": aligned_meas["p50"],
                "target_p50": target_dist["ref_p50"],
                "ssim_before": ssim_b,
                "ssim_after": ssim_a,
                "lpips_before": lpips_b,
                "lpips_after": lpips_a,
                "wd_before": wd_b,
                "wd_after": wd_a,
            })

        df_m = pd.DataFrame(sample_evals)
        mean_err_b = (abs(df_m["orig_mean"] - df_m["target_mean"]) / df_m["target_mean"]).mean() * 100
        mean_err_a = (abs(df_m["aligned_mean"] - df_m["target_mean"]) / df_m["target_mean"]).mean() * 100
        std_err_b = (abs(df_m["orig_std"] - df_m["target_std"]) / (df_m["target_std"] + 1e-4)).mean() * 100
        std_err_a = (abs(df_m["aligned_std"] - df_m["target_std"]) / (df_m["target_std"] + 1e-4)).mean() * 100

        ckpt_results.append({
            "Checkpoint": name,
            "样本数": len(df_m),
            "均值误差(前->后)": f"{mean_err_b:.1f}% -> {mean_err_a:.1f}%",
            "方差误差(前->后)": f"{std_err_b:.1f}% -> {std_err_a:.1f}%",
            "SSIM(前->后)": f"{df_m['ssim_before'].mean():.4f} -> {df_m['ssim_after'].mean():.4f}",
            "SSIM变化": f"{df_m['ssim_after'].mean() - df_m['ssim_before'].mean():+.4f}",
            "LPIPS(前->后)": f"{df_m['lpips_before'].mean():.4f} -> {df_m['lpips_after'].mean():.4f}",
            "LPIPS变化": f"{df_m['lpips_after'].mean() - df_m['lpips_before'].mean():+.4f}",
            "WD距(前->后)": f"{df_m['wd_before'].mean():.2f}px -> {df_m['wd_after'].mean():.2f}px",
            "WD改善": f"{df_m['wd_after'].mean() - df_m['wd_before'].mean():+.2f}px",
        })

    # ==============================================================
    # 步骤 3: 汇总输出 7 个历史 Checkpoint 的完整对比大表
    # ==============================================================
    print("\n" + "=" * 135)
    print("                     【7 个历史 Checkpoint 强分布对齐 (均值/方差/SDF) 完整指标对比大表】")
    print("=" * 135)
    df_all_ckpts = pd.DataFrame(ckpt_results)
    print(df_all_ckpts.to_string(index=False))
    print("=" * 135)


if __name__ == "__main__":
    main()
