import csv, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CSV = "/root/Workspace/xy/DiT/5script/train_top30_clean.csv"
print("exists:", os.path.exists(CSV))
if os.path.exists(CSV):
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("rows:", len(rows))
    print("header:", list(rows[0].keys()))
    p0 = rows[0]["image_path"]
    full = os.path.join("/root/Workspace/xy/DiT", p0)
    print("path0:", p0, "exists:", os.path.isfile(full))
