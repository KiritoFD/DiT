"""全面排查: 所有 csv 里 (书家,书体) pair 不在词表的行。"""
import csv
import glob
import json
import os
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")

m = json.load(open("assets/callig_script_id_map.json", encoding="utf-8"))
pm = set(m["pair_map"].keys())
print(f"  词表: {len(pm)} 个 pair")

FILES = sorted(set(
    glob.glob("assets/eval_*.csv") +
    glob.glob("assets/train_50k_v2*.csv") +
    glob.glob("assets/fs*_*.csv")
))
print(f"  扫描 {len(FILES)} 个 csv\n")

for f in FILES:
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
    except Exception as e:
        print(f"  ✗ {os.path.basename(f)}: {e}")
        continue
    if not rows or "calligrapher_id" not in rows[0] or "script_id" not in rows[0]:
        print(f"  - {os.path.basename(f)}: {len(rows)} 行（无 pair 列，跳过）")
        continue
    bad = [r for r in rows
           if f"{r['calligrapher_id']}:{r['script_id']}" not in pm]
    mark = "★" if bad else " "
    print(f"  {mark} {os.path.basename(f)}: {len(rows)} 行, "
          f"词表外 {len(bad)} 条")
    if bad:
        c = Counter(f"{r['calligrapher']}/{r['script']}"
                    for r in bad)
        print(f"      分布: {dict(c)}")
