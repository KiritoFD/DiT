"""More thorough dirty image scan: check multiple criteria."""
import os, sys, csv
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

CSV_PATH = "5script/train_top6.csv"
IMG_ROOT = "final_imgs_256"

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
print(f"Total rows: {len(rows)}")

def check_one(args):
    idx, path = args
    p = path
    # final_images/xxx.png -> final_imgs_256/xxx.png
    p = p.replace("final_images/", "final_imgs_256/")
    if not os.path.isabs(p):
        p = os.path.join(IMG_ROOT, os.path.basename(p)) if "/" not in p else p
    if not os.path.exists(p):
        return idx, None
    try:
        img = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    except:
        return idx, None
    h, w = img.shape
    # Grey pixels (not pure black/white)
    grey_mask = (img >= 50) & (img <= 205)
    grey_ratio = float(grey_mask.mean())
    mean_val = float(img.mean())
    std_val = float(img.std())
    # Near-white ratio (background)
    white_ratio = float((img > 230).mean())
    # Near-black ratio (strokes)
    black_ratio = float((img < 25).mean())
    # "Dirty" = high grey ratio OR very low std (flat image)
    return idx, grey_ratio, mean_val, std_val, white_ratio, black_ratio, p

tasks = [(i, r["image_path"]) for i, r in enumerate(rows)]
results = []
with ProcessPoolExecutor(max_workers=16) as pool:
    futs = {pool.submit(check_one, t): t[0] for t in tasks}
    done = 0
    for f in as_completed(futs):
        r = f.result()
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(rows)}...", flush=True)
        if r[1] is not None:
            results.append(r)

# Stats
grey_ratios = [r[1] for r in results if r[1] is not None]
means = [r[2] for r in results if r[2] is not None]
stds = [r[3] for r in results if r[3] is not None]
white_ratios = [r[4] for r in results if r[4] is not None]
black_ratios = [r[5] for r in results if r[5] is not None]

print(f"\n=== Statistics ({len(results)} valid images) ===")
print(f"Grey ratio:   mean={np.mean(grey_ratios):.4f} median={np.median(grey_ratios):.4f} max={np.max(grey_ratios):.4f}")
print(f"Mean pixel:   mean={np.mean(means):.1f} median={np.median(means):.1f} min={np.min(means):.1f} max={np.max(means):.1f}")
print(f"Std pixel:    mean={np.mean(stds):.1f} median={np.median(stds):.1f} min={np.min(stds):.1f} max={np.max(stds):.1f}")
print(f"White ratio:  mean={np.mean(white_ratios):.4f} median={np.median(white_ratios):.4f}")
print(f"Black ratio:  mean={np.mean(black_ratios):.4f} median={np.median(black_ratios):.4f}")

# Flag suspicious images
print(f"\n=== Suspicious images ===")
dirty = [r for r in results if r[1] is not None and r[1] > 0.15]
print(f"Grey ratio > 0.15: {len(dirty)}")
for r in sorted(dirty, key=lambda x: -x[1])[:10]:
    print(f"  grey={r[1]:.3f} mean={r[2]:.1f} std={r[3]:.1f} {r[6]}")

flat = [r for r in results if r[3] is not None and r[3] < 10]
print(f"\nStd < 10 (flat/near-empty): {len(flat)}")
for r in sorted(flat, key=lambda x: x[3])[:10]:
    print(f"  std={r[3]:.1f} mean={r[2]:.1f} grey={r[1]:.3f} {r[6]}")

dark = [r for r in results if r[2] is not None and r[2] < 80]
print(f"\nMean < 80 (dark images): {len(dark)}")
for r in sorted(dark, key=lambda x: x[2])[:10]:
    print(f"  mean={r[2]:.1f} std={r[3]:.1f} grey={r[1]:.3f} {r[6]}")
