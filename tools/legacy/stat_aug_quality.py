# -*- coding: utf-8 -*-
"""Sanity check: compare an augmented image vs its source, report pixel diff stats.
Pick a few augmented ids, find their source via aug_meta, compare histograms / SSIM."""
import os, sys, csv, re
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUG_DIR = os.path.join(BASE, "final_imgs_mid_clean")
IMG_DIR = os.path.join(BASE, "final_imgs_256")

meta = []
with open(os.path.join("/tmp/mid_clean_tmp", "aug_meta.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        meta.append(r)
print(f"aug_meta: {len(meta)} rows")

# find rows with same (glyph_id, callig) as some src to compare
# just pick first 5 aug, find their source img_id via train csv
train = list(csv.DictReader(open(os.path.join(BASE, "5script", "train_3top30_common.csv"), encoding="utf-8")))
combo2src = {}
for r in train:
    key = (r["script"], r["character"], r["calligrapher"])
    combo2src.setdefault(key, []).append(int(re.search(r"(\d+)\.png", r["image_path"]).group(1)))

def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize((256, 256), Image.LANCZOS), dtype=np.float32) / 255.0

print("\n=== compare aug vs source (first 8 aug) ===")
for r in meta[:8]:
    aug_id = int(r["new_id"])
    aug_path = os.path.join(AUG_DIR, f"{aug_id}.png")
    key = (r["script"], r["char"], r["calli"])
    src_ids = combo2src.get(key, [])
    if not src_ids:
        continue
    src_id = src_ids[0]
    src_path = os.path.join(IMG_DIR, f"{src_id}.png")
    a = load_gray(aug_path)
    s = load_gray(src_path)
    diff = np.abs(a - s)
    print(f"  aug {aug_id} ({key}) vs src {src_id}: "
          f"mean|diff|={diff.mean():.4f} max={diff.max():.4f} "
          f"ssim~{1 - diff.mean()*2:.3f} ink_ratio aug={ (a<0.5).mean():.3f} src={ (s<0.5).mean():.3f}")

# ink ratio distribution across a sample of aug images
print("\n=== ink ratio distribution (sample 500 aug) ===")
import random
random.seed(0)
sample_ids = [int(r["new_id"]) for r in random.sample(meta, min(500, len(meta)))]
inks = []
for aid in sample_ids:
    a = load_gray(os.path.join(AUG_DIR, f"{aid}.png"))
    inks.append((a < 0.5).mean())
inks = np.array(inks)
print(f"  ink_ratio: mean={inks.mean():.4f} std={inks.std():.4f} "
      f"min={inks.min():.4f} q25={np.percentile(inks,25):.4f} "
      f"q50={np.percentile(inks,50):.4f} q75={np.percentile(inks,75):.4f} max={inks.max():.4f}")
# compare with source ink ratio
src_sample = [combo2src[(r["script"], r["char"], r["calli"])][0]
              for r in random.sample(meta, min(500, len(meta)))
              if (r["script"], r["char"], r["calli"]) in combo2src]
src_inks = [(load_gray(os.path.join(IMG_DIR, f"{sid}.png")) < 0.5).mean() for sid in src_sample[:200]]
src_inks = np.array(src_inks)
print(f"  src ink_ratio: mean={src_inks.mean():.4f} std={src_inks.std():.4f} "
      f"min={src_inks.min():.4f} q50={np.percentile(src_inks,50):.4f} max={src_inks.max():.4f}")
print("DONE")
