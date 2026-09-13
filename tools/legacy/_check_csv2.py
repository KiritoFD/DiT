import csv, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CSV = "/root/Workspace/xy/DiT/5script/train_top30_clean.csv"
with open(CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print("rows:", len(rows))
# check paths
fixed = 0
missing = 0
for r in rows:
    p = r["image_path"]
    if p.startswith("final_images/"):
        p = p.replace("final_images/", "final_imgs_256/", 1)
        fixed += 1
    full = os.path.join("/root/Workspace/xy/DiT", p)
    if not os.path.isfile(full):
        missing += 1
        if missing <= 3:
            print("MISSING:", full)
print(f"fixed: {fixed}, missing: {missing}")
