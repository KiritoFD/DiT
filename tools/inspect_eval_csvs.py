import os
import glob
import pandas as pd

# Let's inspect eval CSVs and samples
csvs = [
    "assets/eval_v13_strict_fixed.csv",
    "assets/eval_seen_v10.csv",
    "assets/eval_fame3_strict_clean_v9.csv",
    "exp-std/csv/eval200fix.csv",
    "exp-std/csv/eval200_fixed.csv"
]
for c in csvs:
    p = os.path.join("/root/Workspace/xy/DiT", c)
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"{c}: {len(df)} rows, columns={list(df.columns)}")
        print(df.head(3))
