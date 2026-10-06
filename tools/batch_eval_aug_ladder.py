#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_eval_aug_ladder.py — 48 机 B/2 增强模型全周期阶段性权重全自动批量评估器

在训练完训 (或手动触发) 释放显存后，自动遍历已落盘的里程碑 checkpoints:
  - 10k 步 (与 S/2, Sp/2 对齐)
  - 20k 步 (与 Sp/2 完训终点对齐)
  - 30k 步
  - 40k 步
  - 48k 步 (10h 终局)
使用 GPU 高速运行 eval_ablation_model.py，产出每个阶段的 Strict SSIM / LPIPS / Skel IoU，
并聚合输出为全周期收敛对比总表。
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

CKPT_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h/checkpoints"
OUT_BASE = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h/evaluations"
PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
EVAL_SCRIPT = (
    "/home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py"
)

os.makedirs(OUT_BASE, exist_ok=True)

# 目标评测阶段 (优先级从高到低)
TARGET_STEPS = [10000, 20000, 30000, 40000, 48000, 5000]


def main():
    print("=" * 80)
    print("【开始 Tier 3 (B/2) 增强模型里程碑权重批量评测流水线】")
    print(f"  权重目录: {CKPT_DIR}")
    print(f"  评测输出: {OUT_BASE}")
    print("=" * 80)

    # 扫描已落盘权重
    available_ckpts = {}
    for pt in glob.glob(os.path.join(CKPT_DIR, "*.pt")):
        fn = os.path.basename(pt)
        nums = re.findall(r"\d+", fn)
        if nums:
            st = int(nums[-1])
            available_ckpts[st] = pt

    print(f"检测到已落盘权重: {sorted(list(available_ckpts.keys()))} 步")

    results_table = []

    for st in TARGET_STEPS:
        if st not in available_ckpts:
            print(f"  [跳过] Step {st} 权重尚未生成或未找到")
            continue

        ckpt_path = available_ckpts[st]
        out_dir = os.path.join(OUT_BASE, f"eval_{st}")
        os.makedirs(out_dir, exist_ok=True)
        metric_file = os.path.join(out_dir, "metrics_40k.json")

        if os.path.exists(metric_file):
            print(f"  [已有缓存] Step {st} 已完成评测，直接读取指标...")
            try:
                with open(metric_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                st_m = m.get("strict", {})
                results_table.append(
                    {
                        "step": st,
                        "strict_ssim": round(st_m.get("ssim_mean", 0), 4),
                        "strict_lpips": round(st_m.get("lpips_mean", 0), 4),
                        "skel_iou": round(st_m.get("skel_iou_mean", 0), 4),
                        "mse": round(st_m.get("mse_mean", 0), 4),
                    }
                )
                continue
            except Exception:
                pass

        print(
            f"\n>>> 正在运行 Step {st} GPU 评测: {ckpt_path} -> {out_dir} ...",
            flush=True,
        )
        t0 = time.time()
        cmd = [PYTHON, "-u", EVAL_SCRIPT, "--ckpt", ckpt_path, "--out-dir", out_dir]
        res = subprocess.run(cmd)
        dt = time.time() - t0

        if res.returncode == 0 and os.path.exists(metric_file):
            print(f"✓ Step {st} 评测成功完成，耗时 {dt:.1f} 秒！", flush=True)
            with open(metric_file, "r", encoding="utf-8") as f:
                m = json.load(f)
            st_m = m.get("strict", {})
            results_table.append(
                {
                    "step": st,
                    "strict_ssim": round(st_m.get("ssim_mean", 0), 4),
                    "strict_lpips": round(st_m.get("lpips_mean", 0), 4),
                    "skel_iou": round(st_m.get("skel_iou_mean", 0), 4),
                    "mse": round(st_m.get("mse_mean", 0), 4),
                }
            )
        else:
            print(f"✗ Step {st} 评测失败 (退出码: {res.returncode})", flush=True)

    # 打印汇总对比表
    print("\n" + "=" * 80)
    print("【Tier 3 (B/2) 增强模型全周期阶段性收敛评测总表】")
    print("=" * 80)
    print(
        f"{'Step':<10} | {'Strict SSIM':<14} | {'Strict LPIPS':<14} | {'Skel IoU':<12} | {'MSE':<10}"
    )
    print("-" * 75)
    results_table.sort(key=lambda x: x["step"])
    for r in results_table:
        print(
            f"{r['step']:<10d} | {r['strict_ssim']:<14.4f} | {r['strict_lpips']:<14.4f} | {r['skel_iou']:<12.4f} | {r['mse']:<10.4f}"
        )
    print("=" * 80)

    # 保存汇总 JSON
    summary_json = os.path.join(OUT_BASE, "aug_ladder_eval_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2, ensure_ascii=False)
    print(f"评测汇总已保存至: {summary_json}")


if __name__ == "__main__":
    main()
