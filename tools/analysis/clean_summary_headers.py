import csv
import pandas as pd
from pathlib import Path

csv_path = Path("assets/aligned_eval_results/all_checkpoints_strong_distribution_summary.csv")
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# Rename columns cleanly
df.columns = [
    "Checkpoint",
    "原均值误差",
    "新均值误差",
    "原方差误差",
    "新方差误差",
    "原P50误差",
    "新P50误差",
    "SSIM变化",
    "LPIPS变化",
    "WD改善",
]

df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("Cleaned CSV headers:")
print(df.to_string(index=False))
