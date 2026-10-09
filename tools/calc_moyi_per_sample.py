import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.stdout.reconfigure(encoding="utf-8")

moyi_dir = "assets/server_48_evals/moyi_12ch_eval200fix"
print("=== 计算 moyi_12ch 全部 187 个样本的 SSIM 与 MSE ===")

moyi_records = []
for idx in range(187):
    gp = os.path.join(moyi_dir, f"g{idx}.png")
    gtp = os.path.join(moyi_dir, f"gt{idx}.png")
    if os.path.exists(gp) and os.path.exists(gtp):
        g_arr = np.array(Image.open(gp).convert("L")).astype(np.float32) / 255.0
        gt_arr = np.array(Image.open(gtp).convert("L")).astype(np.float32) / 255.0
        score_ssim = ssim(g_arr, gt_arr, data_range=1.0)
        score_mse = np.mean((g_arr - gt_arr) ** 2) * 4.0  # 与工程一致的 4x mse
        moyi_records.append({"idx": idx, "ssim": score_ssim, "mse": score_mse})

df_moyi = pd.DataFrame(moyi_records)
print(f"成功计算 {len(df_moyi)} 个样本！")
print(f"  全局均值: SSIM={df_moyi['ssim'].mean():.4f}, MSE={df_moyi['mse'].mean():.4f}")

# 保存为本地 csv
df_moyi.to_csv("assets/moyi_12ch_e200_per_sample.csv", index=False)
