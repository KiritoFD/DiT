#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob
import pandas as pd

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

for name in ["v31_stage1_skel", "v33_stage1_xs", "v32_stage2_img"]:
    p = f"assets/results/{name}/eval_stdskel_batch.csv"
    if not os.path.exists(p):
        print(f"=== {name}: {p} not found ===")
        continue
    df = pd.read_csv(p)
    print(f"\n==================== {name} ====================")
    steps = sorted(df["step"].unique())
    print(f"Total steps: {len(steps)}, min={min(steps)}, max={max(steps)}")
    agg = []
    # Print every 5000 or 10000 steps
    for st in steps:
        if st % 5000 != 0 and st != max(steps):
            continue
        sub = df[df["step"] == st]
        for s in sorted(sub["set"].unique()):
            s_df = sub[sub["set"] == s]
            agg.append({
                "step": st, "set": s, "n": len(s_df),
                "ssim": s_df["ssim"].mean(),
                "mse": s_df["mse"].mean(),
                "lpips": s_df["lpips"].mean(),
                "ink_iou": s_df["ink_iou"].mean()
            })
    print(pd.DataFrame(agg).to_string(index=False))
