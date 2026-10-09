import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("docs/experiments/all_runs_20261005.csv")

# 检查重要运行
long_runs = df[(df["ckpt_last"] >= 5000) & (df["run_dir"].str.contains("202610|v23|v24|v25|v26|v3|v4|v5|v6|v7", na=False))]
print(f"=== 10月及后续长跑实验 ({len(long_runs)} 项) ===")
for r in long_runs[["experiment_name", "ckpt_last", "diffusion_type", "run_dir"]].drop_duplicates("experiment_name").to_dict("records"):
    print(f"• {r['experiment_name']}: ckpt_last={r['ckpt_last']}, type={r['diffusion_type']}, path={r['run_dir']}")
