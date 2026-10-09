import os
import csv
import json
import numpy as np
import pandas as pd

# 1. 首先加载 v68 step 200000 的 187 个样本，建立基于 v68 SSIM 的排序索引
v68_p = "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_stdskel_batch.csv"

def load_e200_from_batch(csv_p, exp_kw=None, step_target=None):
    if not os.path.exists(csv_p):
        return None
    with open(csv_p, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("set") == "eval200fix"]
    if not rows:
        return None
    # 筛选 step
    steps = sorted(list(set(int(r["step"]) for r in rows)))
    target = step_target if step_target in steps else steps[-1]
    rows_target = [r for r in rows if int(r["step"]) == target]
    if len(rows_target) < 187:
        return None
    # 按照 idx 排序
    df = pd.DataFrame(rows_target)
    df["idx"] = df["idx"].astype(int)
    for col in ["ssim", "mse", "ink_ssim", "ink_iou"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df.sort_values(by="idx").reset_index(drop=True), target

df_v68, st_v68 = load_e200_from_batch(v68_p, step_target=200000)
print(f"Loaded v68 step {st_v68}, rows={len(df_v68)}")

# 根据 v68 的 SSIM 确定排序与分位数
df_v68_ranked = df_v68.sort_values(by="ssim", ascending=False).reset_index(drop=True)

# 划分区间:
# Top 25%: 前 47 个样本 (0..46)
# Mid 50%: 中间 93 个样本 (47..139)
# Worst 25%: 尾部 47 个样本 (140..186)
idx_top25 = set(df_v68_ranked.iloc[0:47]["idx"].tolist())
idx_mid50 = set(df_v68_ranked.iloc[47:140]["idx"].tolist())
idx_worst25 = set(df_v68_ranked.iloc[140:187]["idx"].tolist())

print(f"Top 25% 数量: {len(idx_top25)}, Mid 50% 数量: {len(idx_mid50)}, Worst 25% 数量: {len(idx_worst25)}")

# 定义我们要统计的实验集合
candidate_experiments = [
    ("v68_c2ot_200k", "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_stdskel_batch.csv", 200000),
    ("v66_route_150k", "/root/Workspace/xy/DiT/assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_stdskel_batch.csv", 150000),
    ("v65_inj_50k", "/root/Workspace/xy/DiT/assets/results/v65_tables_condinj2468/20261005-231613-v65-tables-condinj2468-adaLN/eval_stdskel_batch.csv", 50000),
    ("v64_skel4ch_40k", "/root/Workspace/xy/DiT/assets/results/v64_skel_tables_4ch/20261005-214434-v64-skel-tables-4ch/eval_stdskel_batch.csv", 40000),
    ("v54_tables_150k", "/root/Workspace/xy/DiT/assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/eval_stdskel_batch.csv", 150000),
    ("v70_stdskel_30k", "/root/Workspace/xy/DiT/assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_stdskel_batch.csv", 30000),
]

summary_stats = []

for name, p, tgt_step in candidate_experiments:
    res = load_e200_from_batch(p, step_target=tgt_step)
    if not res:
        print(f"Warning: could not load {name}")
        continue
    df_exp, st = res
    
    # 划分 4 个子集: all, top25, mid50, worst25
    tiers = {
        "all_187": df_exp,
        "top25": df_exp[df_exp["idx"].isin(idx_top25)],
        "mid50": df_exp[df_exp["idx"].isin(idx_mid50)],
        "worst25": df_exp[df_exp["idx"].isin(idx_worst25)],
    }
    
    exp_summary = {"model": name, "step": st}
    for tname, sub_df in tiers.items():
        exp_summary[f"{tname}_ssim_mean"] = float(sub_df["ssim"].mean())
        exp_summary[f"{tname}_ssim_med"] = float(sub_df["ssim"].median())
        exp_summary[f"{tname}_mse_mean"] = float(sub_df["mse"].mean())
        if "ink_ssim" in sub_df.columns:
            exp_summary[f"{tname}_ink_ssim"] = float(sub_df["ink_ssim"].mean())
    
    summary_stats.append(exp_summary)

df_summary = pd.DataFrame(summary_stats)
out_csv = "/root/Workspace/xy/DiT/exp_milestones/eval200_quartiles_summary.csv"
df_summary.to_csv(out_csv, index=False)
print(f"Saved summary to {out_csv}")
print(df_summary[["model", "step", "all_187_ssim_mean", "top25_ssim_mean", "mid50_ssim_mean", "worst25_ssim_mean"]])
