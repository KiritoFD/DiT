#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/align_eval_to_stroke_table.py —
1. 按照【书法家 × 书体】细粒度统计 GT 真迹并持久化粗细参照表:
   - assets/stroke_width_reference_table.json / .csv
2. 将历史评测模型 (Exp A 70k, Arm 2 50k, Base 55k) 的 eval 图片完全对齐到该参照表 (SDF 亚像素对齐 + 拓扑保护)
3. 落盘对齐后的图片并评测完整指标:
   - Width-P50, P10, P90, Contrast, Weq, Wasserstein 距离
   - SSIM, LPIPS, 连通域拓扑
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
from scipy.ndimage import distance_transform_edt, label
from scipy.stats import wasserstein_distance
from skimage.morphology import medial_axis, skeletonize
from skimage.metrics import structural_similarity as compute_ssim
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "exp-std/csv/eval200_fixed.csv"
REF_TABLE_JSON = ROOT / "assets/stroke_width_reference_table.json"
REF_TABLE_CSV = ROOT / "assets/stroke_width_reference_table.csv"

EVAL_MODELS = {
    "Exp A 70k": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/20261020-065018-Unet-flow/eval_samples_ctrl/step0070000/eval200fix",
    "Arm 2 50k": ROOT / "assets/results/unet_ablation_50k/arm2_writer_film_dec_b768/20261019-105029-Unet-flow/eval_samples_ctrl/step0050000/eval200fix",
    "Base 55k": ROOT / "assets/runs/v83A_55k_correct_eval_20261012/eval_samples_ctrl/step0055000/eval200fix",
}


def measure_stroke_properties(img_np: np.ndarray, threshold: float = 0.5) -> dict:
    """输入归一化灰度图 (1=白底, 0=黑字)，测量笔画几何分布。"""
    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 20:
        return {
            "p10": 0.0, "p50": 0.0, "p90": 0.0,
            "contrast": 1.0, "w_eq": 0.0,
            "widths": np.array([0.0]), "ink_mask": ink_mask,
            "ink_area": 0, "skel_len": 0, "num_fg": 0, "num_holes": 0
        }

    dist_map = distance_transform_edt(ink_mask)
    skel = medial_axis(ink_mask)
    if not skel.any():
        skel = skeletonize(ink_mask)

    if not skel.any():
        return {
            "p10": 0.0, "p50": 0.0, "p90": 0.0,
            "contrast": 1.0, "w_eq": 0.0,
            "widths": np.array([0.0]), "ink_mask": ink_mask,
            "ink_area": int(ink_mask.sum()), "skel_len": 0, "num_fg": 1, "num_holes": 0
        }

    widths = 2.0 * dist_map[skel]
    widths = widths[widths >= 0.5]
    if len(widths) == 0:
        widths = np.array([1.0])

    p10 = float(np.percentile(widths, 10))
    p50 = float(np.percentile(widths, 50))
    p90 = float(np.percentile(widths, 90))
    contrast = p90 / max(p10, 1e-4)

    ink_area = float(ink_mask.sum())
    skel_len = float(skel.sum())
    w_eq = ink_area / max(skel_len, 1.0)

    # 拓扑统计
    _, num_fg = label(ink_mask)
    _, num_bg = label(~ink_mask)
    num_holes = max(0, num_bg - 1)

    return {
        "p10": p10, "p50": p50, "p90": p90,
        "contrast": contrast, "w_eq": w_eq,
        "widths": widths, "ink_mask": ink_mask,
        "ink_area": ink_area, "skel_len": skel_len,
        "num_fg": num_fg, "num_holes": num_holes,
    }


def adjust_stroke_sdf(img_np: np.ndarray, delta: float, threshold: float = 0.5) -> np.ndarray:
    """SDF 亚像素轮廓偏移。"""
    if abs(delta) < 1e-4:
        return img_np

    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 20:
        return img_np

    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    phi_prime = phi + delta
    adjusted_ink = np.clip(phi_prime + 0.5, 0.0, 1.0)
    return (1.0 - adjusted_ink).astype(np.float32)


def align_to_target_width(img_np: np.ndarray, target_p50: float) -> tuple[np.ndarray, float, dict]:
    """对齐单张图像到 target_p50，带有拓扑保护。"""
    orig_meas = measure_stroke_properties(img_np)
    orig_p50 = orig_meas["p50"]
    orig_fg = orig_meas["num_fg"]
    orig_holes = orig_meas["num_holes"]

    if orig_p50 < 1.0 or target_p50 < 1.0:
        return img_np, 0.0, orig_meas

    init_delta = (target_p50 - orig_p50) / 2.0
    init_delta = float(np.clip(init_delta, -1.5, 1.5))

    best_delta = 0.0
    best_img = img_np
    best_meas = orig_meas
    min_diff = abs(orig_p50 - target_p50)

    candidates = [0.0]
    for step in np.linspace(-1.5, 1.5, 13):
        candidates.append(float(step))
    candidates.append(init_delta)
    candidates = sorted(list(set(candidates)))

    for d in candidates:
        cand_img = adjust_stroke_sdf(img_np, d)
        cand_meas = measure_stroke_properties(cand_img)

        # 拓扑保护门禁
        if d < 0 and cand_meas["num_fg"] > orig_fg:  # 变细导致断笔
            continue
        if d > 0 and cand_meas["num_holes"] < orig_holes - 1:  # 加粗导致孔洞粘连消失
            continue

        diff = abs(cand_meas["p50"] - target_p50)
        if diff < min_diff:
            min_diff = diff
            best_delta = d
            best_img = cand_img
            best_meas = cand_meas

    return best_img, best_delta, best_meas


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    print(f"Loaded {len(rows)} samples from {CSV_PATH.name}")

    # ==============================================================
    # 步骤 1: 建立【书法家 × 书体】细粒度粗细参照表
    # ==============================================================
    print("\n[Step 1] Building Reference Table across Calligrapher * Script...")
    slot_gt_measurements = defaultdict(list)

    per_sample_gt = []
    for idx, r in enumerate(rows):
        callig = r["calligrapher"]
        script = r["script"]
        gt_path = ROOT / r["image_path"]

        im_np = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        meas = measure_stroke_properties(im_np)
        meas["idx"] = idx
        meas["calligrapher"] = callig
        meas["script"] = script
        meas["character"] = r["character"]
        meas["gt_path"] = str(gt_path.relative_to(ROOT))
        per_sample_gt.append(meas)

        slot = f"{callig}_{script}"
        slot_gt_measurements[slot].append(meas)

    ref_table = {}
    ref_table_rows = []
    for slot, m_list in sorted(slot_gt_measurements.items()):
        callig, script = slot.split("_")
        all_p10 = [m["p10"] for m in m_list]
        all_p50 = [m["p50"] for m in m_list]
        all_p90 = [m["p90"] for m in m_list]
        all_contrast = [m["contrast"] for m in m_list]
        all_weq = [m["w_eq"] for m in m_list]

        slot_ref = {
            "slot": slot,
            "calligrapher": callig,
            "script": script,
            "sample_count": len(m_list),
            "ref_p10": float(np.mean(all_p10)),
            "ref_p50": float(np.mean(all_p50)),
            "ref_p90": float(np.mean(all_p90)),
            "ref_contrast": float(np.mean(all_contrast)),
            "ref_weq": float(np.mean(all_weq)),
        }
        ref_table[slot] = slot_ref
        ref_table_rows.append(slot_ref)

    REF_TABLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REF_TABLE_JSON, "w", encoding="utf-8") as f:
        json.dump(ref_table, f, indent=2, ensure_ascii=False)

    df_ref = pd.DataFrame(ref_table_rows)
    df_ref.to_csv(REF_TABLE_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved reference table to: {REF_TABLE_JSON.relative_to(ROOT)} and {REF_TABLE_CSV.relative_to(ROOT)}")

    print("\n" + "=" * 95)
    print("                 【细粒度书法家 × 书体 GT 笔画粗细标准参照表】")
    print("=" * 95)
    print(df_ref[[
        "calligrapher", "script", "sample_count",
        "ref_p10", "ref_p50", "ref_p90", "ref_contrast", "ref_weq"
    ]].to_string(index=False))
    print("=" * 95)

    # ==============================================================
    # 步骤 2: 将历史 eval 图片完全对齐到该参照表并评测完整指标
    # ==============================================================
    print("\n[Step 2] Aligning Historical Evaluation Images to Reference Table & Evaluating...")

    from src.eval.decoder import _get_lpips
    lpips_fn = _get_lpips()
    ldev = next(lpips_fn.parameters()).device if lpips_fn else "cpu"
    print(f"LPIPS ready on device: {ldev}")

    model_eval_reports = {}

    for m_name, m_dir in EVAL_MODELS.items():
        if not m_dir.exists():
            print(f"Skipping {m_name}: dir not found")
            continue

        print(f"\n>>> Processing Model: {m_name} ({m_dir.relative_to(ROOT)}) ...")
        aligned_out_dir = m_dir.parent / f"{m_dir.name}_aligned_sdf"
        aligned_out_dir.mkdir(parents=True, exist_ok=True)

        sample_evals = []

        for idx, r in enumerate(rows):
            callig = r["calligrapher"]
            script = r["script"]
            char = r["character"]
            slot = f"{callig}_{script}"

            target_slot_ref = ref_table[slot]
            target_p50 = target_slot_ref["ref_p50"]

            gen_path = m_dir / f"g{idx}.png"
            if not gen_path.exists():
                continue

            gt_meas = per_sample_gt[idx]
            gt_im = np.array(Image.open(ROOT / r["image_path"]).convert("L"), dtype=np.float32) / 255.0
            gen_im = np.array(Image.open(gen_path).convert("L"), dtype=np.float32) / 255.0

            orig_meas = measure_stroke_properties(gen_im)

            # 对齐到该【书家×书体】的细粒度目标粗细 target_p50
            aligned_im, delta, aligned_meas = align_to_target_width(gen_im, target_p50)

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
                "target_p50": target_p50,
                "gt_p50": gt_meas["p50"],
                "orig_p50": orig_meas["p50"],
                "aligned_p50": aligned_meas["p50"],
                "delta": delta,
                "ssim_before": ssim_b,
                "ssim_after": ssim_a,
                "lpips_before": lpips_b,
                "lpips_after": lpips_a,
                "wd_before": wd_b,
                "wd_after": wd_a,
                "orig_weq": orig_meas["w_eq"],
                "aligned_weq": aligned_meas["w_eq"],
                "gt_weq": gt_meas["w_eq"],
                "orig_contrast": orig_meas["contrast"],
                "aligned_contrast": aligned_meas["contrast"],
                "gt_contrast": gt_meas["contrast"],
                "orig_fg": orig_meas["num_fg"],
                "aligned_fg": aligned_meas["num_fg"],
                "gt_fg": gt_meas["num_fg"],
            })

        df_m = pd.DataFrame(sample_evals)
        model_eval_reports[m_name] = df_m

        # 输出该模型汇总
        err_b = (abs(df_m["orig_p50"] - df_m["target_p50"]) / df_m["target_p50"]).mean() * 100
        err_a = (abs(df_m["aligned_p50"] - df_m["target_p50"]) / df_m["target_p50"]).mean() * 100

        print(f"=== {m_name} 对齐前后完整评测 ===")
        print(f"  P50 距目标误差:   {err_b:.2f}% -> {err_a:.2f}% (收敛 {err_b - err_a:.2f}%)")
        print(f"  Wasserstein 距离: {df_m['wd_before'].mean():.2f} px -> {df_m['wd_after'].mean():.2f} px (减少 {df_m['wd_before'].mean() - df_m['wd_after'].mean():.2f} px)")
        print(f"  SSIM 结构相似度:  {df_m['ssim_before'].mean():.4f} -> {df_m['ssim_after'].mean():.4f} ({df_m['ssim_after'].mean() - df_m['ssim_before'].mean():+.4f})")
        print(f"  LPIPS 感知误差:   {df_m['lpips_before'].mean():.4f} -> {df_m['lpips_after'].mean():.4f} ({df_m['lpips_after'].mean() - df_m['lpips_before'].mean():+.4f})")
        print(f"  对齐图像落盘目录: {aligned_out_dir.relative_to(ROOT)}")

    # ==============================================================
    # 步骤 3: 汇总各模型对比大表与细粒度切片大表
    # ==============================================================
    print("\n" + "=" * 110)
    print("                     【全模型粗细对齐前后完整指标对比总览表 (187 样本)】")
    print("=" * 110)
    macro_rows = []
    for m_name, df_m in model_eval_reports.items():
        err_b = (abs(df_m["orig_p50"] - df_m["target_p50"]) / df_m["target_p50"]).mean() * 100
        err_a = (abs(df_m["aligned_p50"] - df_m["target_p50"]) / df_m["target_p50"]).mean() * 100
        macro_rows.append({
            "模型": m_name,
            "原始 P50": f"{df_m['orig_p50'].mean():.2f}px",
            "对齐 P50": f"{df_m['aligned_p50'].mean():.2f}px",
            "P50误差(前->后)": f"{err_b:.1f}% -> {err_a:.1f}%",
            "平均Delta": f"{df_m['delta'].mean():+.2f}px",
            "SSIM(前->后)": f"{df_m['ssim_before'].mean():.4f} -> {df_m['ssim_after'].mean():.4f}",
            "SSIM变化": f"{df_m['ssim_after'].mean() - df_m['ssim_before'].mean():+.4f}",
            "LPIPS(前->后)": f"{df_m['lpips_before'].mean():.4f} -> {df_m['lpips_after'].mean():.4f}",
            "WD(前->后)": f"{df_m['wd_before'].mean():.2f}px -> {df_m['wd_after'].mean():.2f}px",
        })
    print(pd.DataFrame(macro_rows).to_string(index=False))
    print("=" * 110)

    # 导出细粒度对比表格 (Exp A 70k 的分 slot 明细)
    df_exp_a = model_eval_reports["Exp A 70k"]
    slot_summary = []
    for slot, g in df_exp_a.groupby("slot"):
        callig, script = slot.split("_")
        slot_summary.append({
            "书法家": callig,
            "书体": script,
            "样本数": len(g),
            "目标P50": f"{g['target_p50'].mean():.2f}px",
            "原始P50": f"{g['orig_p50'].mean():.2f}px",
            "对齐P50": f"{g['aligned_p50'].mean():.2f}px",
            "均值Delta": f"{g['delta'].mean():+.2f}px",
            "SSIM前": f"{g['ssim_before'].mean():.4f}",
            "SSIM后": f"{g['ssim_after'].mean():.4f}",
            "SSIM益": f"{g['ssim_after'].mean() - g['ssim_before'].mean():+.4f}",
            "LPIPS益": f"{g['lpips_before'].mean() - g['lpips_after'].mean():+.4f}",
            "WD改善": f"{g['wd_after'].mean() - g['wd_before'].mean():+.2f}px",
        })
    print("\n" + "=" * 115)
    print("             【Exp A 70k: 分【书法家 × 书体】细粒度粗细对齐评测明细表】")
    print("=" * 115)
    print(pd.DataFrame(slot_summary).to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    main()
