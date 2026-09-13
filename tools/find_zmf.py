# -*- coding: utf-8 -*-
"""find zhao mengfu variants in all fame csvs."""
import csv
import glob

variants = ["赵孟頫", "趙孟頫", "赵孟俯", "趙孟俯", "赵孟虎", "赵孟"]
for path in glob.glob("assets/*.csv") + glob.glob("assets/**/*.csv", recursive=True):
    try:
        with open(path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames
            if not cols or "calligrapher" not in cols:
                continue
            counts = {}
            for row in r:
                name = row["calligrapher"]
                for v in variants:
                    if v in name:
                        counts[name] = counts.get(name, 0) + 1
            if counts:
                print(f"{path}: {counts}")
    except Exception:
        pass
print("done")
