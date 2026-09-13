"""Diagnose: are the GT images the correct images, or are they from wrong indices?
Check if gt{i}.png in eval_samples matches the eval CSV row i."""
import os, sys, csv, json
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

# Load eval CSV
rows = list(csv.DictReader(open("5script/eval500_clean.csv", encoding="utf-8")))
print(f"eval CSV: {len(rows)} rows")

# Check a few GT images against original files
img_root = "final_imgs_256"
step_dir = "5script/results/s10_b4_grey_clear/20260824-151758-s10-b4-grey-clear/checkpoints/eval_samples/step0001000"

for i in [0, 1, 2, 100, 200]:
    # GT saved by eval
    gt_path = os.path.join(step_dir, f"gt{i}.png")
    if not os.path.exists(gt_path):
        print(f"  [{i}] gt{i}.png missing")
        continue
    gt_saved = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0

    # Original file from CSV
    csv_row = rows[i]
    orig_path = csv_row["image_path"]
    if not os.path.isabs(orig_path):
        orig_path = os.path.join(img_root, orig_path)
    if not os.path.exists(orig_path):
        print(f"  [{i}] original {orig_path} missing")
        continue
    orig_img = np.asarray(Image.open(orig_path).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0

    mse = float(np.mean((gt_saved - orig_img) ** 2))
    print(f"  [{i}] csv={csv_row['image_path'][:50]} | saved_gt mean={gt_saved.mean():.3f} orig mean={orig_img.mean():.3f} | MSE(gt vs orig)={mse:.6f}")

# Also check sample vs gt
print("\n--- Sample vs GT analysis ---")
for i in [0, 1, 2, 100, 200]:
    sp = os.path.join(step_dir, f"sample{i}.png")
    gp = os.path.join(step_dir, f"gt{i}.png")
    if not os.path.exists(sp) or not os.path.exists(gp):
        continue
    s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
    g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0

    # Per-channel MSE
    for ch, name in enumerate(["R", "G", "B"]):
        ch_mse = float(np.mean((s[:,:,ch] - g[:,:,ch]) ** 2))
        print(f"  [{i}] {name}: sample mean={s[:,:,ch].mean():.3f} gt mean={g[:,:,ch].mean():.3f} MSE={ch_mse:.4f}")
    print()
