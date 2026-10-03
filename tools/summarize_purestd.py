import pandas as pd
import numpy as np

csv_path = "/root/Workspace/xy/DiT/exp-std/runs_purestd/eval_stdskel_batch.csv"
try:
    df = pd.read_csv(csv_path)
    print("=" * 60)
    print(f"【runs_purestd (p=1.0 纯标准字条件) 评测汇总】 总记录: {len(df)} 行")
    print("=" * 60)
    num_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in ['step', 'idx']]
    summary = df.groupby('step')[num_cols].mean()
    print(summary.to_string())
except Exception as e:
    print(f"Error reading {csv_path}: {e}")
