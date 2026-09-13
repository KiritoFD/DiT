"""Definitive MSE comparison: old eval (eval_gen.py on [-1,1]) vs new daemon ([0,1]).
Also verify the old eval approach on a saved PNG pair."""
import os, sys, glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

step_dir = "5script/results/s10_b4_grey_clear/20260824-151758-s10-b4-grey-clear/checkpoints/eval_samples/step0001000"

# Method 1: new daemon — [0,1] pixel MSE
print("=== Method 1: new daemon [0,1] MSE ===")
mses_01 = []
for i in range(50):
    sp = os.path.join(step_dir, f"sample{i}.png")
    gp = os.path.join(step_dir, f"gt{i}.png")
    if not (os.path.exists(sp) and os.path.exists(gp)):
        continue
    s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
    g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
    mses_01.append(float(np.mean((s - g) ** 2)))
print(f"  [0,1] MSE: mean={np.mean(mses_01):.5f} (n={len(mses_01)})")

# Method 2: old eval_gen.py — [-1,1] pixel MSE (F.mse_loss on [-1,1] tensors)
print("\n=== Method 2: old eval_gen [-1,1] MSE (F.mse_loss) ===")
mses_11 = []
for i in range(50):
    sp = os.path.join(step_dir, f"sample{i}.png")
    gp = os.path.join(step_dir, f"gt{i}.png")
    if not (os.path.exists(sp) and os.path.exists(gp)):
        continue
    s_np = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
    g_np = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
    # [0,1] -> [-1,1]
    s_t = torch.from_numpy(s_np.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1
    g_t = torch.from_numpy(g_np.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1
    mses_11.append(float(F.mse_loss(s_t, g_t).item()))
print(f"  [-1,1] MSE: mean={np.mean(mses_11):.5f} (n={len(mses_11)})")

# Method 3: [0,255] pixel MSE (what some old code might have used)
print("\n=== Method 3: [0,255] MSE ===")
mses_255 = []
for i in range(50):
    sp = os.path.join(step_dir, f"sample{i}.png")
    gp = os.path.join(step_dir, f"gt{i}.png")
    if not (os.path.exists(sp) and os.path.exists(gp)):
        continue
    s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32)
    g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32)
    mses_255.append(float(np.mean((s - g) ** 2)))
print(f"  [0,255] MSE: mean={np.mean(mses_255):.2f} (n={len(mses_255)})")

print(f"\n=== Summary ===")
print(f"  [0,1] MSE = {np.mean(mses_01):.5f}  (new daemon)")
print(f"  [-1,1] MSE = {np.mean(mses_11):.5f}  (old eval_gen F.mse_loss)")
print(f"  [0,255] MSE = {np.mean(mses_255):.2f}  (raw pixel)")
print(f"  Ratio [-1,1]/[0,1] = {np.mean(mses_11)/np.mean(mses_01):.1f}x")
print(f"  Ratio [0,255]/[0,1] = {np.mean(mses_255)/np.mean(mses_01):.0f}x")
