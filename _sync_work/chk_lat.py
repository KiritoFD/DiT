"""量落盘 latent 的统计: 生成器吐的东西量级对不对, 一测就知道。
正常 VAE latent (×0.18215) 的 std 约 1, |max| 约 3~5。若 |max| 几十上百 -> 采样发散
-> vae.decode 饱和 -> poster 全白 (mean=1.000)。
"""
import glob

import numpy as np

for d in ("assets/results/v34_e2e/predskel_step005000/seen20",
          "assets/results/v34_e2e/predskel_step005000/strict84",
          "data/top10_style23/shards_std",
          "data/top10_style23/shards_gtskel_w7"):
    fs = sorted(glob.glob(f"{d}/shard_*.npz"))
    if not fs:
        print(f"{d}: 无文件")
        continue
    z = np.load(fs[0])
    x = np.asarray(z["latents"], np.float32)
    print(f"{d}\n  keys={list(z.keys())} shape={x.shape} dtype={z['latents'].dtype} "
          f"mean={x.mean():.3f} std={x.std():.3f} |max|={abs(x).max():.2f} "
          f"nan={int(np.isnan(x).sum())} zero_frac={(x == 0).mean():.3f}")
