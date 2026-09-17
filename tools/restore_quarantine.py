# -*- coding: utf-8 -*-
"""restore_quarantine.py — 回滚 clean_base_dataset.py 的剪切 (按 manifest 恢复)。"""
import csv
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Q = "data/_quarantine"
MAN = os.path.join(Q, "_manifest.csv")

n = 0
if os.path.exists(MAN):
    for r in csv.DictReader(open(MAN, encoding="utf-8")):
        op = r["orig_path"]
        parts = op.split("/")
        q = os.path.join(Q, parts[2] if len(parts) > 2 else "misc",
                         os.path.basename(op))
        if os.path.exists(q):
            d = os.path.dirname(op)
            if d:
                os.makedirs(d, exist_ok=True)
            shutil.move(q, op)
            n += 1
print(f"restored base {n}")

m = 0
for f in glob.glob(os.path.join(Q, "base_sym", "*.png")):
    os.makedirs("data/imgs/base_sym", exist_ok=True)
    shutil.move(f, os.path.join("data/imgs/base_sym", os.path.basename(f)))
    m += 1
print(f"restored aug {m}")
print("done")
