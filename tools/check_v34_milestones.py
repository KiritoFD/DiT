#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import pandas as pd

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

p = "exp/v34_stage2_mix25/eval_stdskel_batch.csv"
if not os.path.exists(p):
    print(f"Not found: {p}")
    sys.exit(0)

df = pd.read_csv(p)
steps = sorted(df["step"].unique())
print(f"=== v34_stage2_mix25 全量进度看板 (已跑步数: {steps}) ===")

agg = []
for st in steps:
    sub = df[df["step"] == st]
    for s_name in ["seen", "strict", "seen_pred", "strict_pred"]:
        s_df = sub[sub["set"] == s_name]
        if len(s_df) == 0:
            continue
        agg.append({
            "step": st,
            "set": s_name,
            "n": len(s_df),
            "ssim": s_df["ssim"].mean(),
            "ssim_med": s_df["ssim"].median(),
            "mse": s_df["mse"].mean(),
            "lpips": s_df["lpips"].mean(),
            "ink_ssim": s_df["ink_ssim"].mean(),
            "frag": s_df["frag_ratio"].mean() if "frag_ratio" in s_df else None
        })

df_res = pd.DataFrame(agg)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
print(df_res.to_string(index=False))
