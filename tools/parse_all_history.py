import os
import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# 1. 分析 00_exhaustive_experiment_chronicle.md
chronicle_path = "docs/archive/20261005_slim/04_experiments/00_exhaustive_experiment_chronicle.md"
with open(chronicle_path, "r", encoding="utf-8") as f:
    text = f.read()

epochs = re.findall(r"^## (第.+)$", text, flags=re.MULTILINE)
print(f"=== 编年史中记录的纪元 ({len(epochs)} 个) ===")
for ep in epochs:
    print(" ", ep)

experiments = re.findall(r"^### 🧪 实验档案：`([^`]+)`", text, flags=re.MULTILINE)
print(f"\n编年史中详细建档的实验数量: {len(experiments)} 项")

# 2. 分析 all_runs_20261005.csv 中的所有运行
csv_path = "docs/experiments/all_runs_20261005.csv"
df = pd.read_csv(csv_path)
print(f"\n=== all_runs_20261005.csv 全量统计 ===")
print(f"总运行记录数: {len(df)}")
unique_exps = df["experiment_name"].dropna().unique()
print(f"独立实验系列数: {len(unique_exps)}")

# 查看 9 月底到 10 月初之后的最新运行
recent_runs = df[df["run_dir"].str.contains("202610", na=False)]
print(f"\n2026年10月最新运行数: {len(recent_runs)}")
for r in recent_runs[["experiment_name", "ckpt_last", "diffusion_type", "run_dir"]].drop_duplicates("experiment_name").to_dict("records"):
    print(f"  {r['experiment_name']}: ckpt_last={r['ckpt_last']}, type={r['diffusion_type']}, path={r['run_dir'][:60]}")
