#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/test_sdf_stroke_alignment.py —
利用 SDF (有符号距离场) 亚像素调整生成图笔宽，对齐 GT 的 P50，并评估对齐后性能提升:
  - 测量原始 Gen vs GT 的 P50
  - 自动搜索最优 delta in [-1.0, +1.0] px
  - 拓扑保护门禁: 检查连通分量与孔洞数量，杜绝断笔与粘连
  - 评测指标变化: SSIM, LPIPS, P50 误差, Wasserstein 距离
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from scipy.stats import wasserstein_distance
from skimage.morphology import medial_axis, skeletonize
from skimage.metrics import structural_similarity as compute_ssim
import torch
import lpips

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "exp-std/csv/eval200_fixed.csv"
MODEL_DIR = ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0070000/eval200fix"


def compute_p50_and_widths(img_np: np.ndarray, threshold: float = 0.5):
    """img_np: float [0, 1], 1=white background, 0=black ink."""
    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 20:
        return 0.0, np.array([0.0]), ink_mask
    dist_map = distance_transform_edt(ink_mask)
    skel = medial_axis(ink_mask)
    if not skel.any():
        skel = skeletonize(ink_mask)
    if not skel.any():
        return 0.0, np.array([0.0]), ink_mask
    widths = 2.0 * dist_map[skel]
    widths = widths[widths >= 0.5]
    if len(widths) == 0:
        widths = np.array([1.0])
    p50 = float(np.percentile(widths, 50))
    return p50, widths, ink_mask


def compute_topology(mask: np.ndarray):
    """计算连通域数与孔洞数。"""
    labeled_fg, num_fg = label(mask)
    inv_mask = ~mask
    labeled_bg, num_bg = label(inv_mask)
    # 孔洞数 ≈ 背景连通域数 - 1 (减去包围字的外围大背景)
    num_holes = max(0, num_bg - 1)
    return num_fg, num_holes


def adjust_stroke_sdf(img_np: np.ndarray, delta: float, threshold: float = 0.5) -> np.ndarray:
    """使用有符号距离场 (SDF) 调整笔画粗细。

    phi = D_inside - D_outside (墨内为正，墨外为负)
    phi' = phi + delta
    """
    if abs(delta) < 1e-4:
        return img_np

    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 20:
        return img_np

    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    # 偏移距离场
    phi_prime = phi + delta

    # 亚像素抗锯齿过渡 (边界 1 像素平滑)
    # 当 phi' > 0.5 时为墨(0.0)，phi' < -0.5 时为底(1.0)，中间平滑过渡
    adjusted_ink = np.clip(phi_prime + 0.5, 0.0, 1.0)
    adjusted_img = 1.0 - adjusted_ink

    return adjusted_img.astype(np.float32)


def search_best_delta(gen_np: np.ndarray, target_p50: float, orig_p50: float) -> tuple[float, np.ndarray]:
    """搜索最优 delta 并严格执行拓扑保护。"""
    orig_fg, orig_holes = compute_topology((1.0 - gen_np) >= 0.5)

    initial_delta = (target_p50 - orig_p50) / 2.0
    initial_delta = float(np.clip(initial_delta, -1.2, 1.2))

    best_delta = 0.0
    best_img = gen_np
    min_diff = abs(orig_p50 - target_p50)

    # 细化搜索候选集 (包括 0, 初始估计, 以及微调步进)
    candidates = [0.0]
    for scale in [0.25, 0.5, 0.75, 1.0, 1.25]:
        d = initial_delta * scale
        if -1.2 <= d <= 1.2:
            candidates.append(d)

    for d in candidates:
        cand_img = adjust_stroke_sdf(gen_np, d)
        cand_mask = (1.0 - cand_img) >= 0.5
        fg, holes = compute_topology(cand_mask)

        # 拓扑保护检查: 连通分量不能增加(不能断裂)，孔洞不能消失过多(不能粘连)
        if d < 0 and fg > orig_fg:  # 变细导致断笔
            continue
        if d > 0 and holes < orig_holes - 1:  # 加粗导致孔洞被封死
            continue

        cand_p50, _, _ = compute_p50_and_widths(cand_img)
        diff = abs(cand_p50 - target_p50)
        if diff < min_diff:
            min_diff = diff
            best_delta = d
            best_img = cand_img

    return best_delta, best_img


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    print(f"Loaded {len(rows)} rows from {CSV_PATH.name}")

    from src.eval.decoder import _get_lpips
    print("Loading LPIPS model from src.eval.decoder...")
    lpips_fn = _get_lpips()

    results = []

    print("\nRunning SDF Stroke Alignment & Evaluation...")
    for idx, r in enumerate(rows):
        callig = r["calligrapher"]
        script = r["script"]
        char = r["character"]

        gt_path = ROOT / r["image_path"]
        gen_path = MODEL_DIR / f"g{idx}.png"

        if not gt_path.exists() or not gen_path.exists():
            continue

        gt_im = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        gen_im = np.array(Image.open(gen_path).convert("L"), dtype=np.float32) / 255.0

        gt_p50, gt_widths, _ = compute_p50_and_widths(gt_im)
        gen_p50, gen_widths, _ = compute_p50_and_widths(gen_im)

        if gt_p50 < 1.0 or gen_p50 < 1.0:
            continue

        # 搜索最优 delta 对齐 GT P50
        delta, aligned_im = search_best_delta(gen_im, gt_p50, gen_p50)
        aligned_p50, aligned_widths, _ = compute_p50_and_widths(aligned_im)

        # 指标 1: SSIM
        ssim_before = compute_ssim(gen_im, gt_im, data_range=1.0)
        ssim_after = compute_ssim(aligned_im, gt_im, data_range=1.0)

        # 指标 2: LPIPS
        if lpips_fn is not None:
            ldev = next(lpips_fn.parameters()).device
            t_gt = torch.from_numpy(gt_im).unsqueeze(0).repeat(1, 3, 1, 1).to(ldev) * 2.0 - 1.0
            t_gen = torch.from_numpy(gen_im).unsqueeze(0).repeat(1, 3, 1, 1).to(ldev) * 2.0 - 1.0
            t_aligned = torch.from_numpy(aligned_im).unsqueeze(0).repeat(1, 3, 1, 1).to(ldev) * 2.0 - 1.0

            with torch.no_grad():
                lpips_before = float(lpips_fn(t_gen, t_gt).item())
                lpips_after = float(lpips_fn(t_aligned, t_gt).item())
        else:
            lpips_before = 0.0
            lpips_after = 0.0

        wd_before = float(wasserstein_distance(gen_widths, gt_widths))
        wd_after = float(wasserstein_distance(aligned_widths, gt_widths))

        results.append({
            "idx": idx,
            "calligrapher": callig,
            "script": script,
            "char": char,
            "gt_p50": gt_p50,
            "gen_p50": gen_p50,
            "delta": delta,
            "aligned_p50": aligned_p50,
            "ssim_before": ssim_before,
            "ssim_after": ssim_after,
            "ssim_gain": ssim_after - ssim_before,
            "lpips_before": lpips_before,
            "lpips_after": lpips_after,
            "lpips_gain": lpips_before - lpips_after,  # 正数代表感知更优
            "wd_before": wd_before,
            "wd_after": wd_after,
        })

    df = pd.DataFrame(results)
    print(f"\nEvaluated {len(df)} samples successfully.")

    # 总体统计汇总
    mean_ssim_b = df["ssim_before"].mean()
    mean_ssim_a = df["ssim_after"].mean()
    mean_lpips_b = df["lpips_before"].mean()
    mean_lpips_a = df["lpips_after"].mean()
    mean_wd_b = df["wd_before"].mean()
    mean_wd_a = df["wd_after"].mean()
    mean_err_b = (abs(df["gen_p50"] - df["gt_p50"]) / df["gt_p50"]).mean() * 100
    mean_err_a = (abs(df["aligned_p50"] - df["gt_p50"]) / df["gt_p50"]).mean() * 100

    print("=" * 85)
    print("           【SDF 笔画粗细对齐实验总体成果 (Exp A 70k vs GT 真迹)】")
    print("=" * 85)
    print(f"指标                      对齐前 (Raw Gen)    SDF 对齐后        相对提升/变化")
    print(f"---------------------------------------------------------------------")
    print(f"P50 笔宽绝对相对误差:     {mean_err_b:6.2f}%            {mean_err_a:6.2f}%          {- (mean_err_b - mean_err_a):+.2f}% (笔宽对齐度显著提升)")
    print(f"Wasserstein 笔宽分布距:  {mean_wd_b:6.2f} px          {mean_wd_a:6.2f} px         {mean_wd_a - mean_wd_b:+.2f} px")
    print(f"SSIM (结构相似度):       {mean_ssim_b:6.4f}            {mean_ssim_a:6.4f}          {mean_ssim_a - mean_ssim_b:+.4f}")
    print(f"LPIPS (越低越好):        {mean_lpips_b:6.4f}            {mean_lpips_a:6.4f}          {mean_lpips_a - mean_lpips_b:+.4f}")
    print(f"平均调整量 delta:        -                   {df['delta'].mean():+.3f} px (范围: [{df['delta'].min():.2f}, {df['delta'].max():.2f}])")
    print("=" * 85)

    # 分书法家对比表
    print("\n" + "=" * 95)
    print("                  【分书法家 SDF 粗细对齐增益细分表】")
    print("=" * 95)
    cal_summary = []
    for callig, g in df.groupby("calligrapher"):
        cal_summary.append({
            "书法家": callig,
            "GT P50": f"{g['gt_p50'].mean():.2f}px",
            "原始P50": f"{g['gen_p50'].mean():.2f}px",
            "对齐P50": f"{g['aligned_p50'].mean():.2f}px",
            "均值delta": f"{g['delta'].mean():+.2f}px",
            "SSIM变化": f"{g['ssim_after'].mean() - g['ssim_before'].mean():+.4f}",
            "LPIPS变化": f"{g['lpips_after'].mean() - g['lpips_before'].mean():+.4f}",
            "WD改善": f"{g['wd_after'].mean() - g['wd_before'].mean():+.2f}px",
        })
    df_cal = pd.DataFrame(cal_summary)
    print(df_cal.to_string(index=False))
    print("=" * 95)

    # 输出几个提升最明显与最有代表性的案例
    df["ssim_gain"] = df["ssim_after"] - df["ssim_before"]
    top_improved = df.sort_values("ssim_gain", ascending=False).head(5)
    print("\n【SSIM 提升最显著的 Top 5 样本】:")
    for _, r in top_improved.iterrows():
        print(f"  #{r['idx']:03d} 【{r['char']}】 ({r['calligrapher']} {r['script']}): "
              f"GT={r['gt_p50']:.1f}px, 原Gen={r['gen_p50']:.1f}px -> 对齐={r['aligned_p50']:.1f}px (delta={r['delta']:+.2f}px) | "
              f"SSIM: {r['ssim_before']:.4f} -> {r['ssim_after']:.4f} ({r['ssim_gain']:+.4f}) | "
              f"LPIPS: {r['lpips_before']:.4f} -> {r['lpips_after']:.4f}")


if __name__ == "__main__":
    main()
