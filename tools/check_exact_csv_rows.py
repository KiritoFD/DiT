import os
import csv
import json

def check_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"=== {path} (总行数: {len(rows)}) ===")
    for i, r in enumerate(rows[:10]):
        ch = r.get("character") or r.get("char")
        cal = r.get("calligrapher")
        sc = r.get("script")
        img_p = r.get("image_path")
        print(f"  [{i:02d}] char={ch} | callig={cal} | script={sc} | img={img_p}")
    return rows

check_csv("/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv")
print()
check_csv("/home/ds/Workspace/moyi/exp-std-csv/eval200.csv")
print()
check_csv("/home/ds/Workspace/moyi/assets/eval_top10_real_200.csv")
