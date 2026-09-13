"""Deep MSE diagnosis: compare GT-vs-GT, GT-vs-constant, sample-vs-GT distributions."""
import os, sys, csv, glob
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

step_dir = "5script/results/s10_b4_grey_clear/20260824-151758-s10-b4-grey-clear/checkpoints/eval_samples/step0001000"

# Load all GTs
print("Loading all GTs...")
gts = []
for i in range(455):
    gp = os.path.join(step_dir, f"gt{i}.png")
    if os.path.exists(gp):
        gts.append(np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0)
print(f"Loaded {len(gts)} GTs")

# GT statistics
gt_means = [g.mean() for g in gts]
print(f"GT mean: avg={np.mean(gt_means):.3f} std={np.std(gt_means):.3f} min={np.min(gt_means):.3f} max={np.max(gt_means):.3f}")

# MSE between random pairs of GTs (should be the "noise floor")
print("\n--- GT-vs-GT MSE (random pairs) ---")
np.random.seed(42)
gt_gt_mses = []
for _ in range(200):
    i, j = np.random.choice(len(gts), 2, replace=False)
    gt_gt_mses.append(float(np.mean((gts[i] - gts[j]) ** 2)))
gt_gt_mses = np.array(gt_gt_mses)
print(f"GT-GT MSE: mean={gt_gt_mses.mean():.4f} median={np.median(gt_gt_mses):.4f} std={gt_gt_mses.std():.4f}")

# MSE between sample and GT
print("\n--- Sample-vs-GT MSE ---")
sg_mses = []
for i in range(min(200, len(gts))):
    sp = os.path.join(step_dir, f"sample{i}.png")
    if os.path.exists(sp):
        s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
        sg_mses.append(float(np.mean((s - gts[i]) ** 2)))
sg_mses = np.array(sg_mses)
print(f"Sample-GT MSE: mean={sg_mses.mean():.4f} median={np.median(sg_mses):.4f} std={sg_mses.std():.4f}")

# MSE between sample and a constant image (mean of GT)
print("\n--- Sample-vs-constant (GT mean) MSE ---")
avg_gt = np.mean(gts, axis=0)
sc_mses = []
for i in range(min(200, len(gts))):
    sp = os.path.join(step_dir, f"sample{i}.png")
    if os.path.exists(sp):
        s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
        sc_mses.append(float(np.mean((s - avg_gt) ** 2)))
sc_mses = np.array(sc_mses)
print(f"Sample-const MSE: mean={sc_mses.mean():.4f}")

# Check: are sample and GT actually aligned (same index)?
print("\n--- Alignment check: sample{i} vs gt{j} for various i,j ---")
for i in [0, 1, 2]:
    sp = os.path.join(step_dir, f"sample{i}.png")
    if not os.path.exists(sp): continue
    s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
    # Find which GT index gives minimum MSE
    best_mse, best_j = 1e9, -1
    for j in range(min(50, len(gts))):
        m = float(np.mean((s - gts[j]) ** 2))
        if m < best_mse:
            best_mse, best_j = m, j
    print(f"  sample{i}: best match gt{best_j} (MSE={best_mse:.4f}), same-index MSE={float(np.mean((s-gts[i])**2)):.4f}")

# Check step 15000 too
print("\n=== step 15000 ===")
step2 = "5script/results/s10_b4_grey_clear/20260824-164822-s10-b4-grey-clear/checkpoints/eval_samples/step0015000"
sg2 = []
for i in range(min(200, 455)):
    sp = os.path.join(step2, f"sample{i}.png")
    gp = os.path.join(step2, f"gt{i}.png")
    if os.path.exists(sp) and os.path.exists(gp):
        s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        sg2.append(float(np.mean((s - g) ** 2)))
sg2 = np.array(sg2)
print(f"step15000 Sample-GT MSE: mean={sg2.mean():.4f} median={np.median(sg2):.4f} std={sg2.std():.4f}")
print(f"step15000 frac < 0.1: {(sg2<0.1).mean():.2%}")
print(f"step15000 frac < 0.05: {(sg2<0.05).mean():.2%}")
