"""Check pixel value distributions of sample vs gt images to diagnose MSE."""
import os, sys, glob
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = "/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear"
# Check multiple steps
for run_dir in sorted(glob.glob(os.path.join(base, "*/checkpoints/eval_samples/step*"))):
    step = os.path.basename(run_dir)
    print(f"\n=== {step} ===")
    for i in [0, 1, 2, 100, 200]:
        sp = os.path.join(run_dir, f"sample{i}.png")
        gp = os.path.join(run_dir, f"gt{i}.png")
        if not (os.path.exists(sp) and os.path.exists(gp)):
            continue
        s = np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0
        g = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32) / 255.0
        mse = float(np.mean((s - g) ** 2))
        print(f"  [{i}] sample: mean={s.mean():.3f} min={s.min():.3f} max={s.max():.3f} | "
              f"gt: mean={g.mean():.3f} min={g.min():.3f} max={g.max():.3f} | "
              f"MSE={mse:.5f}")
        if i == 0:
            # Check if inverted
            inv_mse = float(np.mean((1.0 - s - g) ** 2))
            print(f"       inverted MSE (1-s vs g)={inv_mse:.5f}")
    break  # just first step for now
