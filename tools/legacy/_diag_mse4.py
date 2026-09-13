"""Step 15000 alignment check + baseline comparison."""
import os, sys, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

step2 = "5script/results/s10_b4_grey_clear/20260824-164822-s10-b4-grey-clear/checkpoints/eval_samples/step0015000"

# Load first 50 GTs
gts = []
for i in range(50):
    gp = os.path.join(step2, f"gt{i}.png")
    if os.path.exists(gp):
        gts.append(np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0)
    else:
        gts.append(None)

print("=== step 15000 alignment ===")
for i in [0, 1, 2, 10, 20]:
    sp = os.path.join(step2, f"sample{i}.png")
    if not os.path.exists(sp) or gts[i] is None:
        continue
    s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
    same_mse = float(np.mean((s - gts[i]) ** 2))
    best_mse, best_j = 1e9, -1
    for j in range(len(gts)):
        if gts[j] is None: continue
        m = float(np.mean((s - gts[j]) ** 2))
        if m < best_mse:
            best_mse, best_j = m, j
    print(f"  sample{i}: same-index({i}) MSE={same_mse:.4f} | best match gt{best_j} MSE={best_mse:.4f}")

# Check: what would MSE be if we just output a white image?
print("\n=== baselines ===")
white_mses = []
for i in range(50):
    if gts[i] is None: continue
    white = np.ones_like(gts[i])
    white_mses.append(float(np.mean((white - gts[i]) ** 2)))
print(f"white-image MSE: {np.mean(white_mses):.4f}")

# Check: MSE of GT with itself (should be 0)
g0 = gts[0]
print(f"GT self-MSE: {float(np.mean((g0 - g0)**2)):.6f}")

# Check: are sample and gt the same size?
s0 = np.asarray(Image.open(os.path.join(step2, "sample0.png")).convert("RGB"))
g0_img = np.asarray(Image.open(os.path.join(step2, "gt0.png")).convert("RGB"))
print(f"\nsample0 shape: {s0.shape}, gt0 shape: {g0_img.shape}")

# Check unique pixel values in a GT (is it truly grayscale stored as RGB?)
print(f"\ngt0 unique values (first 20): {np.unique(g0_img.flatten())[:20]}")
print(f"gt0 R==G? {np.all(g0_img[:,:,0] == g0_img[:,:,1])}, G==B? {np.all(g0_img[:,:,1] == g0_img[:,:,2])}")
