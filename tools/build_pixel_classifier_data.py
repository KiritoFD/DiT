"""Build a pixel-level glyph classifier image dataset from MCCD.

Maps 3top30 img_ids to local MCCD images via archive/final_manifest.json,
resizes to 128x128 (lighter than 256), saves to a compact npz for training.

Output: _classifier_pixel_data.npz
  - images: (N, 3, 128, 128) float16
  - img_ids: (N,) int64
  - glyph_ids: (N,) int64 (raw, will remap at train time)
"""
import os, sys, csv, json, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')
os.environ["XFORMERS_DISABLED"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MCCD_CHAR = os.path.join(ROOT, "MCCD", "MCCD", "MCCD_Character", "character_dataset")
MANIFEST = os.path.join(ROOT, "archive", "final_manifest.json")
TRAIN_CSV = os.path.join(ROOT, "5script", "train_3top30_nobeike.csv")
OUT = os.path.join(ROOT, "_classifier_pixel64_data.npz")
SIZE = 64

print("Loading manifest...")
manifest = json.load(open(MANIFEST, encoding='utf-8'))
id2entry = {e['img_id']: e for e in manifest}
print(f"  manifest: {len(id2entry)} entries")

# Collect needed img_ids from train_3top30_nobeike.csv
needed = {}
with open(TRAIN_CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        iid = int(os.path.basename(r['image_path']).replace('.png', ''))
        needed[iid] = int(r['glyph_id'])
print(f"  needed img_ids: {len(needed)}")

# Build images
images = []
img_ids = []
glyph_ids = []
missing = 0
errors = 0
for i, (iid, gid) in enumerate(needed.items()):
    entry = id2entry.get(iid)
    if not entry:
        missing += 1
        continue
    op = entry['orig_path']  # e.g. train/㐁/㐁-印-宋-广韵-161207.png
    basename = os.path.basename(op)
    char = entry['orig_char']
    local = os.path.join(MCCD_CHAR, char, basename)
    try:
        img = Image.open(local).convert("L").resize((SIZE, SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float16) / 127.5 - 1.0  # [-1, 1]
        images.append(arr)
        img_ids.append(iid)
        glyph_ids.append(gid)
    except Exception as e:
        errors += 1
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(needed)} processed, {missing} missing, {errors} errors")

images = np.stack(images).astype(np.float16)  # (N, 128, 128)
images = images[:, None, :, :]  # (N, 1, 128, 128)  — grayscale, 1 channel
img_ids = np.array(img_ids, dtype=np.int64)
glyph_ids = np.array(glyph_ids, dtype=np.int64)
print(f"\nDone: {len(images)} images, {missing} missing, {errors} errors")
print(f"  images shape: {images.shape}, dtype: {images.dtype}")
print(f"  size: {images.nbytes/1e6:.1f} MB raw")

np.savez_compressed(OUT, images=images, img_ids=img_ids, glyph_ids=glyph_ids)
print(f"  saved: {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
