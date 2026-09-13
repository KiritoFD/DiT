import csv, os, sys, json, numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "/root/Workspace/xy/DiT"
CSV = os.path.join(BASE, "5script", "train_top30_clean.csv")
IMG_ROOT = BASE

def fix_path(p):
    if p.startswith("final_images/"):
        return p.replace("final_images/", "final_imgs_256/", 1)
    return p

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"CSV rows: {len(rows)}")

def analyze_one(r):
    p = os.path.join(IMG_ROOT, fix_path(r["image_path"]))
    try:
        img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        grey = float(((arr > 0.15) & (arr < 0.85)).mean())
        return grey
    except Exception:
        return None

print("analyzing all images...")
with ThreadPoolExecutor(max_workers=32) as pool:
    greys = list(pool.map(analyze_one, rows))

greys_valid = [g for g in greys if g is not None]
greys_arr = np.array(greys_valid)
print(f"valid: {len(greys_valid)}")
for thresh in [0.1, 0.2, 0.3, 0.5]:
    n = (greys_arr > thresh).sum()
    print(f"  grey > {thresh}: {n} ({100*n/len(greys_arr):.2f}%)")

# Find the worst 20
worst = np.argsort(greys_arr)[::-1][:20]
print("\nworst 20:")
for idx in worst:
    r = rows[idx]
    print(f"  {r['image_path']} grey={greys_arr[idx]:.3f} char={r['character']} script={r['script']}")
