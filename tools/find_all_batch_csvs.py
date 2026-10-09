import os
import pandas as pd
import glob

# 检查 4090 上的各个实验目录下的 eval_stdskel_batch.csv
base = "/root/Workspace/xy/DiT/assets/results"
batch_csvs = glob.glob(f"{base}/**/eval_stdskel_batch.csv", recursive=True)
print(f"找到 {len(batch_csvs)} 个 eval_stdskel_batch.csv:")
for p in batch_csvs:
    df = pd.read_csv(p)
    sets = df["set"].unique() if "set" in df.columns else []
    steps = df["step"].unique() if "step" in df.columns else []
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
    print(f"  [{exp_name}] sets={sets}, max_step={max(steps) if len(steps) else 'none'}, total_rows={len(df)}")
