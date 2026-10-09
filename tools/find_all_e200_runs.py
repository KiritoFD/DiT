import os
import glob
import csv
import pandas as pd

base = "/root/Workspace/xy/DiT/assets/results"

# 我们寻找在 eval200fix 上有完整记录的实验
experiments = {}

for root, dirs, files in os.walk(base):
    if "eval_stdskel_batch.csv" in files:
        p = os.path.join(root, "eval_stdskel_batch.csv")
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # 过滤 set == eval200fix
            e200_rows = [r for r in rows if r.get("set") == "eval200fix"]
            if len(e200_rows) >= 187:
                # 找到最大 step
                steps = sorted(list(set(int(r["step"]) for r in e200_rows)))
                max_step = steps[-1]
                latest_rows = [r for r in e200_rows if int(r["step"]) == max_step]
                exp_id = os.path.basename(os.path.dirname(os.path.dirname(p)))
                run_id = os.path.basename(os.path.dirname(p))
                key = f"{exp_id}_{max_step}"
                experiments[key] = {
                    "exp_id": exp_id,
                    "run_id": run_id,
                    "step": max_step,
                    "rows": latest_rows,
                    "batch_csv": p
                }
        except Exception as e:
            pass

print(f"找到 {len(experiments)} 个拥有完整 eval200fix 的实验版本:")
for k, v in sorted(experiments.items()):
    print(f"  {k}: {len(v['rows'])} 行 (exp_id: {v['exp_id']}, step: {v['step']})")
