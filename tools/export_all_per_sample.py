import os
import csv
import pandas as pd

base = "/root/Workspace/xy/DiT"
out_csv = "/root/Workspace/xy/DiT/exp_milestones/eval200fix_models_per_sample.csv"

models_info = [
    ("v68", "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_stdskel_batch.csv", 200000),
    ("v66", "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_stdskel_batch.csv", 150000),
    ("v54", "assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/eval_stdskel_batch.csv", 150000),
    ("v70", "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_stdskel_batch.csv", 30000),
]

all_rows = []

for mname, rel_p, step in models_info:
    csv_p = os.path.join(base, rel_p)
    if os.path.exists(csv_p):
        with open(csv_p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("set") == "eval200fix" and int(r.get("step", 0)) == step:
                    r["model_name"] = mname
                    all_rows.append(r)

df = pd.DataFrame(all_rows)
print(f"导出 {len(df)} 行数据")
df.to_csv(out_csv, index=False)
print(f"已保存到 {out_csv}")
