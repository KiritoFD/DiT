#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob, json
import pandas as pd

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

stage1_dirs = [d for d in glob.glob("assets/results/*stage1*") if os.path.isdir(d)]
print(f"Found stage1 directories: {stage1_dirs}")

for s_dir in stage1_dirs:
    exp_name = os.path.basename(s_dir)
    print(f"\n==================== {exp_name} ====================")
    
    # 1. Check resolved config
    cfgs = glob.glob(f"{s_dir}/*/resolved_config.json")
    if cfgs:
        cfg = json.load(open(cfgs[0], encoding="utf-8"))
        print(f"Model: {cfg.get('model')}")
        print(f"Target Latent Shards (训练目标): {cfg.get('latent_shards_dir')}")
        print(f"Condition Shards (输入条件): {cfg.get('skel_latent_shards_dir')}")
        print(f"Eval Skel Shards: {cfg.get('eval_skel_latent_shards_dir')}")
        print(f"Data CSV: {cfg.get('data_csv')}")
        print(f"Max steps: {cfg.get('max_steps')}, Batch size: {cfg.get('global_batch_size', cfg.get('batch_size'))}")
    
    # 2. Check batch / summary eval results
    p_batch = f"{s_dir}/eval_stdskel_batch.csv"
    if os.path.exists(p_batch):
        df = pd.read_csv(p_batch)
        steps = sorted(df["step"].unique())
        print(f"Evaluated steps: {len(steps)} (min={min(steps)}, max={max(steps)})")
        
        # Latest step metrics
        latest = max(steps)
        sub = df[df["step"] == latest]
        print(f"\n--- Latest Step {latest} Standalone Metrics (预测骨架 vs 真实目标) ---")
        for s in sorted(sub["set"].unique()):
            s_df = sub[sub["set"] == s]
            print(f"  [{s:<10}] (n={len(s_df):>2}): SSIM={s_df['ssim'].mean():.4f} (med={s_df['ssim'].median():.4f}) | MSE={s_df['mse'].mean():.5f} | LPIPS={s_df['lpips'].mean():.4f} | Ink_IoU={s_df['ink_iou'].mean():.4f}")
            
        # Key checkpoints progression
        print(f"\n--- Progression History (Strict) ---")
        strict_prog = []
        for st in steps:
            s_df = df[(df["step"] == st) & (df["set"] == "strict")]
            if len(s_df):
                strict_prog.append({
                    "step": st,
                    "ssim": round(s_df["ssim"].mean(), 4),
                    "mse": round(s_df["mse"].mean(), 5),
                    "lpips": round(s_df["lpips"].mean(), 4),
                    "ink_iou": round(s_df["ink_iou"].mean(), 4)
                })
        print(pd.DataFrame(strict_prog).to_string(index=False))
