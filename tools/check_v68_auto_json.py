import os
import json
import pandas as pd

json_path = "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/checkpoints/eval_auto_200000.json"
print("=== 1. 查看 eval_auto_200000.json ===")
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("keys:", list(data.keys()) if isinstance(data, dict) else len(data))
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                print(f"  {k}: list of length {len(v)}, sample={v[:3]}")
            else:
                print(f"  {k}: {v}")

csv_batch = "/root/Workspace/xy/DiT/assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_stdskel_batch.csv"
print("\n=== 2. 查看 eval_stdskel_batch.csv ===")
if os.path.exists(csv_batch):
    df = pd.read_csv(csv_batch)
    print(f"rows: {len(df)}, cols: {df.columns.tolist()[:10]}")
    print(df.tail(3))
