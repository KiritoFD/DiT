# -*- coding: utf-8 -*-
import os, sys, csv, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

rows = list(csv.DictReader(open("assets/eval_fame_strict_clean.csv", encoding="utf-8")))
print(f"eval rows: {len(rows)}")
clean = [r for r in rows if "clean" in r["image_path"]]
print(f"clean rows: {len(clean)}")
missing = []
for r in clean:
    p = r["image_path"]
    if not os.path.isfile(p):
        missing.append(p)
print(f"missing clean eval files: {len(missing)}")
for m in missing[:5]:
    print(f"  MISSING: {m}")
# 所有 eval 图存在性
miss_all = [r["image_path"] for r in rows if not os.path.isfile(r["image_path"])]
print(f"all eval rows missing files: {len(miss_all)}")
# 打印几个 clean 样例
for r in clean[:3]:
    print(f"  clean: {r['image_path']}")
