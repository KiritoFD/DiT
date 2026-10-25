import pandas as pd
from pathlib import Path

csv_p = Path("assets/results/unet_subsets/u100_base_top5_noaug/20261020-195826-Unet-flow/eval_stdskel_summary.csv")
if csv_p.exists():
    df = pd.read_csv(csv_p)
    cols = [c for c in ["step", "set", "n", "ssim_mean", "ssim_med", "lpips_mean", "mse_mean", "soft_ink_mean", "ink_ssim_mean"] if c in df.columns]
    print(df[cols].to_string(index=False))
else:
    print("CSV not found:", csv_p)
