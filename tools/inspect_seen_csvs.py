import os
import csv
import json

base = "/root/Workspace/xy/DiT"
seen_csvs = [
    "assets/eval_v13_seen.csv",
    "assets/eval_seen_v10.csv",
    "exp-std/csv/eval_seen_20.csv",
    "assets/eval_top10_seen_20.csv"
]

for cp in seen_csvs:
    p = os.path.join(base, cp)
    if os.path.exists(p):
        print(f"=== {cp} ===")
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        print(f"  Rows count: {len(rows)}")
        for i, r in enumerate(rows[:5]):
            print(f"  [{i}] char={r.get('character') or r.get('char')}, callig={r.get('calligrapher')}, script={r.get('script')}")
