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

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

DEFAULT_CKPT_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug_v66route"
DEFAULT_PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
DEFAULT_EVAL_SCRIPT = "/home/ds/Workspace/DiT/experiments/ablation_phase_a/eval_ablation_model.py"


def parse_args():
    parser = argparse.ArgumentParser(description="全自动里程碑权重批量评估流水线")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Checkpoints 所在目录 (若未指定，自动搜索最新运行目录)")
    parser.add_argument("--out-base", type=str, default=None,
                        help="评测指标输出根目录 (默认在 ckpt_dir 同级 evaluations)")
    parser.add_argument("--steps", type=str, default="5000,10000,20000,30000,40000,48000",
                        help="目标评测步数列表 (逗号分隔)")
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON,
                        help="Python 解释器路径")
    parser.add_argument("--eval-script", type=str, default=DEFAULT_EVAL_SCRIPT,
                        help="eval_ablation_model.py 路径")
    return parser.parse_args()


def resolve_ckpt_dir(target_dir):
    if target_dir and os.path.exists(target_dir):
        # 如果传入的是直接包含 .pt 的 checkpoints 目录
        if any(f.endswith(".pt") for f in os.listdir(target_dir)):
            return target_dir
        # 如果传入的是 run 目录，检查子目录 checkpoints
        sub = os.path.join(target_dir, "checkpoints")
        if os.path.exists(sub):
            return sub

    # 自动搜索 DEFAULT_CKPT_DIR 下最新的 run
    if os.path.exists(DEFAULT_CKPT_DIR):
        runs = sorted(glob.glob(os.path.join(DEFAULT_CKPT_DIR, "*v66*")))
        if runs:
            latest_run = runs[-1]
            sub = os.path.join(latest_run, "checkpoints")
            if os.path.exists(sub):
                return sub
            return latest_run

    # 回退至旧版 tier3_b_aug
    fallback = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h/checkpoints"
    return fallback


def main():
    args = parse_args()
    ckpt_dir = resolve_ckpt_dir(args.ckpt_dir)
    if not args.out_base:
        parent = os.path.dirname(ckpt_dir.rstrip("/"))
        out_base = os.path.join(parent, "evaluations")
    else:
        out_base = args.out_base
    os.makedirs(out_base, exist_ok=True)

    target_steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]

    print("=" * 80)
    print("【开始里程碑权重全周期批量评测流水线】")
    print(f"  权重目录: {ckpt_dir}")
    print(f"  评测输出: {out_base}")
    print(f"  目标步数: {target_steps}")
    print("=" * 80)

    # 扫描已落盘权重
    available_ckpts = {}
    for pt in glob.glob(os.path.join(ckpt_dir, "*.pt")):
        fn = os.path.basename(pt)
        nums = re.findall(r"\d+", fn)
        if nums:
            st = int(nums[-1])
            available_ckpts[st] = pt

    print(f"检测到已落盘权重: {sorted(list(available_ckpts.keys()))} 步")

    results_table = []

    for st in target_steps:
        if st not in available_ckpts:
            print(f"  [跳过] Step {st} 权重尚未生成或未找到")
            continue

        ckpt_path = available_ckpts[st]
        out_dir = os.path.join(out_base, f"eval_{st}")
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
        cmd = [args.python, "-u", args.eval_script, "--ckpt", ckpt_path, "--out-dir", out_dir]
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
