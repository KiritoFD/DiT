import csv, os, sys, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
EVAL_CSV = os.path.join(BASE, "5script", "eval500_clean.csv")

def fix_path(p):
    if p.startswith("final_images/"):
        return p.replace("final_images/", "final_imgs_256/", 1)
    return p

rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
print(f"eval500 rows: {len(rows)}")

results = []
for i, r in enumerate(rows):
    p = os.path.join(BASE, fix_path(r["image_path"]))
    try:
        img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        grey = float(((arr > 0.15) & (arr < 0.85)).mean())
    except Exception:
        grey = 1.0
    results.append((i, r["image_path"], grey, r["character"], r["script"]))

results.sort(key=lambda x: -x[2])
print("\ntop 15 dirty in eval500:")
for idx, path, grey, char, script in results[:15]:
    print(f"  [{idx}] {path} grey={grey:.3f} char={char} script={script}")

dirty3 = [r for r in results if r[2] > 0.3]
dirty2 = [r for r in results if r[2] > 0.2]
print(f"\ngrey>0.3: {len(dirty3)}, grey>0.2: {len(dirty2)}")
