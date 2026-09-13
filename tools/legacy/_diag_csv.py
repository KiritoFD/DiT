"""Check the eval CSV format and image_path values."""
import csv, os
os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("5script/eval500_clean.csv", encoding="utf-8")))
print("CSV columns:", list(rows[0].keys()))
for r in rows[:5]:
    print(f"  image_path={r['image_path']!r}")
print(f"\nTotal rows: {len(rows)}")

# Check if path has double prefix
img_root = "final_imgs_256"
p0 = rows[0]["image_path"]
print(f"\nFirst path: {p0!r}")
print(f"Starts with img_root? {p0.startswith(img_root)}")
full = os.path.join(img_root, p0) if not p0.startswith(img_root) else p0
print(f"Full path: {full!r}")
print(f"Exists? {os.path.exists(full)}")

# Try without double prefix
if p0.startswith(img_root + "/"):
    alt = p0[len(img_root)+1:]
    alt_full = os.path.join(img_root, alt)
    print(f"\nAlt path (stripped): {alt_full!r}")
    print(f"Exists? {os.path.exists(alt_full)}")
