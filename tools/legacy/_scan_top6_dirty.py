"""Scan top6 train CSV for dirty grey images (大面积灰色).
Same criteria as top30 clean: remove images where grey ratio > 0.2.
Grey ratio = fraction of pixels in grey range [50, 205] (not black, not white)."""
import os, sys, csv, glob
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

CSV_PATH = "5script/train_top6.csv"
IMG_ROOT = "final_images"
GREY_LO, GREY_HI = 50, 205
GREY_THRESH = 0.2

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
print(f"Total rows: {len(rows)}")

def check_one(args):
    idx, path = args
    p = path
    if not os.path.isabs(p):
        p = os.path.join(IMG_ROOT, p)
    if not os.path.exists(p):
        return idx, None, None, None
    try:
        img = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    except Exception:
        return idx, None, None, None
    grey_mask = (img >= GREY_LO) & (img <= GREY_HI)
    grey_ratio = float(grey_mask.mean())
    mean_val = float(img.mean())
    return idx, grey_ratio, mean_val, p

tasks = [(i, r["image_path"]) for i, r in enumerate(rows)]
dirty = []
with ProcessPoolExecutor(max_workers=16) as pool:
    futs = {pool.submit(check_one, t): t[0] for t in tasks}
    done = 0
    for f in as_completed(futs):
        idx, grey_ratio, mean_val, p = f.result()
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(rows)}...", flush=True)
        if grey_ratio is not None and grey_ratio > GREY_THRESH:
            dirty.append((idx, p, grey_ratio, mean_val))

print(f"\nDirty (grey_ratio > {GREY_THRESH}): {len(dirty)} / {len(rows)} ({100*len(dirty)/len(rows):.1f}%)")
if dirty:
    print("\nTop 20 dirtiest:")
    dirty.sort(key=lambda x: -x[2])
    for idx, p, gr, mv in dirty[:20]:
        print(f"  [{idx}] {p} grey_ratio={gr:.3f} mean={mv:.1f}")

    # Save dirty list
    with open("5script/train_top6_dirty.csv", "w", encoding="utf-8") as f:
        f.write("image_path,grey_ratio,mean_val\n")
        for idx, p, gr, mv in dirty:
            f.write(f"{p},{gr:.4f},{mv:.2f}\n")
    print(f"\nSaved to 5script/train_top6_dirty.csv")
