#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/compare_interventions.py — 精确对比 SkelNet、第一次加数据、架构接口修复的量化收益"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

v21_p = "assets/results/v21_skelnet_200k/eval_stdskel_summary.csv"
stdaug_p = "assets/results/std_callig_aug/eval_stdskel_summary.csv"
v23_p = "assets/results/v23_splitnorm/eval_stdskel_summary.csv"

df_v21 = pd.read_csv(v21_p)
df_std = pd.read_csv(stdaug_p)
df_v23 = pd.read_csv(v23_p)

steps = [10000, 25000, 50000, 75000]

print("=== 严格未见生僻字零样本泛化 (Strict SSIM, 249字) 同步数对比 ===")
print(f"{'Step':<8} | {'v21 (50k真迹底库)':<18} | {'std_callig_aug (加15k字库)':<26} | {'v23_splitnorm (接口独立LN)':<26}")
print("-" * 80)
for st in steps:
    s_v21 = df_v21[(df_v21['step']==st) & (df_v21['set']=='strict')]['ssim_mean'].values
    s_std = df_std[(df_std['step']==st) & (df_std['set']=='strict')]['ssim_mean'].values
    s_v23 = df_v23[(df_v23['step']==st) & (df_v23['set']=='strict')]['ssim_mean'].values

    v21_val = f"{s_v21[0]:.4f}" if len(s_v21) else "N/A"
    std_val = f"{s_std[0]:.4f}" if len(s_std) else "N/A"
    v23_val = f"{s_v23[0]:.4f}" if len(s_v23) else "N/A"
    print(f"{st:<8} | {v21_val:<18} | {std_val:<26} | {v23_val:<26}")
print("-" * 80)

print("\n=== 风格特异度 (Target Specificity, tgt_spec) 同步数对比 ===")
print(f"{'Step':<8} | {'v21 (50k真迹底库)':<18} | {'std_callig_aug (加15k字库)':<26} | {'v23_splitnorm (接口独立LN)':<26}")
print("-" * 80)
for st in steps:
    s_v21 = df_v21[(df_v21['step']==st) & (df_v21['set']=='strict')]['tgt_spec'].values
    s_std = df_std[(df_std['step']==st) & (df_std['set']=='strict')]['tgt_spec'].values
    s_v23 = df_v23[(df_v23['step']==st) & (df_v23['set']=='strict')]['tgt_spec'].values

    v21_val = f"+{s_v21[0]:.4f}" if len(s_v21) else "N/A"
    std_val = f"+{s_std[0]:.4f}" if len(s_std) else "N/A"
    v23_val = f"+{s_v23[0]:.4f}" if len(s_v23) else "N/A"
    print(f"{st:<8} | {v21_val:<18} | {std_val:<26} | {v23_val:<26}")
print("-" * 80)
