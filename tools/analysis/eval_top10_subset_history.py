#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/eval_top10_subset_history.py — 从历史逐样本 batch.csv 中反查 Top 10 样本的历史表现"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

EVAL_STRICT = "assets/eval_v13_strict_fixed.csv"
df_eval = pd.read_csv(EVAL_STRICT)

# 找出 Top 10 书家
top10_cals = ["王羲之", "苏轼", "赵孟頫", "欧阳询", "颜真卿", "褚遂良", "米芾", "柳公权", "何绍基", "文徵明"]

top10_indices = df_eval[df_eval["calligrapher"].isin(top10_cals)].index.tolist()
print(f"Top 10 在 strict 集中的样本下标数: {len(top10_indices)} 个")

models = {
    "v13_base_50k": "assets/results/v13_base_50k/eval_stdskel_batch.csv",
    "v21_skelnet_200k": "assets/results/v21_skelnet_200k/eval_stdskel_batch.csv",
    "std_callig_aug": "assets/results/std_callig_aug/eval_stdskel_batch.csv",
    "v23_splitnorm": "assets/results/v23_splitnorm/eval_stdskel_batch.csv"
}

steps = [25000, 50000, 75000, 150000]

print("\n=== 历史模型在【Top 10 专属严格未见生僻字 (84样本)】上的表现切片 ===")
print(f"{'Step':<8} | {'v13 Base':<14} | {'v21 SkelNet':<14} | {'std_aug (+15k)':<16} | {'v23 splitnorm':<16}")
print("-" * 75)

for st in steps:
    row_strs = []
    for m_name, p in models.items():
        if not os.path.exists(p):
            row_strs.append("     N/A     ")
            continue
        df_b = pd.read_csv(p)
        sub = df_b[(df_b["step"] == st) & (df_b["set"] == "strict") & (df_b["idx"].isin(top10_indices))]
        if len(sub) > 0:
            ssim_m = sub["ssim"].mean()
            row_strs.append(f"{ssim_m:.4f} (n={len(sub)})")
        else:
            row_strs.append("     N/A     ")
    print(f"{st:<8} | {' | '.join(row_strs)}")

print("-" * 75)
