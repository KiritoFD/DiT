import sys, os
ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch as th
import numpy as np
from src.eval.metrics import ssim_torch, frag_ratio

c = th.load('/root/Workspace/xy/DiT/data/top10_style23/eval_real200_cache.pt', map_location='cpu')
std = c['std_pngs']
gt = c['gt_pngs']

ssim_raw = ssim_torch(std, gt).numpy()
print(f"【输入标准字 vs 真实真迹】SSIM 均值: {np.mean(ssim_raw):.4f}, 中位: {np.median(ssim_raw):.4f}")

std_gray = std.mean(dim=1).numpy()
gt_gray = gt.mean(dim=1).numpy()
frags = [frag_ratio(std_gray[i:i+1], gt_gray[i:i+1]) for i in range(len(std_gray))]
print(f"【输入标准字 vs 真实真迹】frag_ratio 均值: {np.mean(frags):.4f}, 中位: {np.median(frags):.4f}")
