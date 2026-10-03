import os, sys
import pandas as pd
import numpy as np
import glob

print("=================================================================")
print("【远程新实验全景审计：exp-std 目录全量分析】")
print("=================================================================")

# 1. 查看 exp-std 下的所有子目录和 runs
print("\n--- 1. exp-std/ 结构目录 ---")
for p in sorted(glob.glob("exp-std/*")):
    print(f"  {p}")

print("\n--- 2. exp-std/runs/ 下的历史运行 ---")
for p in sorted(glob.glob("exp-std/runs/*")):
    print(f"  {p}")

print("\n--- 3. exp-std/runs_5050/ 的当前评测数据 (按步数聚合) ---")
csv_path = "exp-std/runs_5050/eval_stdskel_batch.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"总记录数: {len(df)} 行, 字段: {df.columns.tolist()}")
    
    num_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in ['step', 'idx']]
    summary = df.groupby('step')[num_cols].mean()
    print(summary.to_string())
else:
    print(f"未找到 {csv_path}")

print("\n--- 4. 检查当前训练日志末尾 ---")
active_dirs = sorted(glob.glob("exp-std/runs_5050/*-v46*"))
if active_dirs:
    log_file = os.path.join(active_dirs[-1], "log.txt")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            print("".join(lines[-15:]))
