#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob
import numpy as np
import pandas as pd
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

print("=== 深度审查 Stage 1 独立骨架预测性能 ===")

# 1. 检查两个 stage1 实验的历史指标
for exp_name in ["v31_stage1_skel", "v33_stage1_xs"]:
    p_batch = f"assets/results/{exp_name}/eval_stdskel_batch.csv"
    if not os.path.exists(p_batch):
        print(f"File not found: {p_batch}")
        continue
    df = pd.read_csv(p_batch)
    print(f"\n==================== {exp_name} ====================")
    latest_step = df["step"].max()
    print(f"最新评估步数: {latest_step}")
    for s_name in ["seen", "strict"]:
        sub = df[(df["step"] == latest_step) & (df["set"] == s_name)]
        if len(sub) == 0:
            continue
        print(f"[{s_name:<6}] n={len(sub):>2} | SSIM={sub['ssim'].mean():.4f} (中位={sub['ssim'].median():.4f}) | MSE={sub['mse'].mean():.5f} | LPIPS={sub['lpips'].mean():.4f} | Ink_IoU={sub['ink_iou'].mean():.4f}")

# 2. 检查生成的骨架图像 (eval_samples_ctrl)
# v31_stage1_skel latest step vs v33_stage1_xs latest step
for exp_name in ["v31_stage1_skel", "v33_stage1_xs"]:
    latest_step_dir = sorted(glob.glob(f"assets/results/{exp_name}/eval_samples_ctrl/step*"))[-1]
    step_num = os.path.basename(latest_step_dir)
    print(f"\n[{exp_name}] 最新生成样本目录: {step_num}")
    
    # 检查 strict 下的样本 0, 1, 5
    for sub in ["strict", "g"]:
        dir_p = os.path.join(latest_step_dir, sub)
        if os.path.exists(dir_p):
            sample_files = glob.glob(f"{dir_p}/g*.png")
            print(f"  子目录 {sub}: 包含 {len(sample_files)} 个生成骨架")
            if sample_files:
                im0 = Image.open(os.path.join(dir_p, "g0.png")).convert("L")
                arr0 = np.asarray(im0)
                print(f"    g0.png 像素统计: min={arr0.min()}, max={arr0.max()}, mean={arr0.mean():.1f}, 黑色墨迹比={(arr0<128).mean()*100:.2f}%")
