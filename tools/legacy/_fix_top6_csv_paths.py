"""Normalize image_path column in top6 CSVs: final_images/<id>.png -> final_imgs_256/<id>.png
so prepare_eval_cache's img_root join works (matches the older eval CSVs' format)."""
import csv, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def fix(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    changed = 0
    for r in rows:
        p = r["image_path"]
        if "final_images/" in p:
            r["image_path"] = p.replace("final_images/", "final_imgs_256/")
            changed += 1
    if changed:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"{path}: {len(rows)} rows, {changed} paths rewritten")
    return rows[0]["image_path"] if rows else None

for p in ["5script/eval100_top6.csv", "5script/show2_top6.csv", "5script/seen2_top6.csv"]:
    sample = fix(p)
    print(f"  sample: {sample}")