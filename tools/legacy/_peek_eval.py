import csv, sys, os, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
# 当前 eval 集
ev = list(csv.DictReader(open(os.path.join(BASE, "5script", "eval100_top30_clean.csv"), encoding="utf-8")))
print(f"current eval: {len(ev)} rows")
print(f"columns: {list(ev[0].keys())}")
# script 分布
from collections import Counter
sc = Counter(r["script"] for r in ev)
print("script 分布:", dict(sc))

# 看原始 eval (扩展前) 有多少
for f in ["eval100_top30.csv", "eval.csv", "test5.csv", "final_test.csv", "final_eval.csv"]:
    p = os.path.join(BASE, "5script", f)
    if not os.path.exists(p):
        p = os.path.join(BASE, f)
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        print(f"{f}: {len(rows)} rows, cols={list(rows[0].keys())[:4]}")
