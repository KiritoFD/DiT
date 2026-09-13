"""Check if sample0 is literally a copy of gt12 (alignment/indexing bug)."""
import os, sys, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

d = "5script/results/s10_b4_grey_clear/20260824-164822-s10-b4-grey-clear/checkpoints/eval_samples/step0015000"

s0 = np.asarray(Image.open(os.path.join(d, "sample0.png")).convert("RGB"))
g0 = np.asarray(Image.open(os.path.join(d, "gt0.png")).convert("RGB"))
g12 = np.asarray(Image.open(os.path.join(d, "gt12.png")).convert("RGB"))

print(f"sample0 == gt0?  {np.array_equal(s0, g0)}  diff={np.abs(s0.astype(int)-g0.astype(int)).mean():.1f}")
print(f"sample0 == gt12? {np.array_equal(s0, g12)} diff={np.abs(s0.astype(int)-g12.astype(int)).mean():.1f}")

# Also: is sample0 == gt0 at the pixel level but shifted/transformed?
# Check raw byte comparison
import hashlib
for name in ["sample0", "gt0", "gt12", "sample1", "gt1"]:
    data = open(os.path.join(d, f"{name}.png"), "rb").read()
    print(f"{name}: {hashlib.md5(data).hexdigest()[:16]} size={len(data)}")

# Check several samples against several GTs
print("\n--- Cross-check: sample{i} vs gt{j} exact match ---")
for i in range(10):
    si = np.asarray(Image.open(os.path.join(d, f"sample{i}.png")).convert("RGB"))
    for j in range(20):
        gj = np.asarray(Image.open(os.path.join(d, f"gt{j}.png")).convert("RGB"))
        if np.array_equal(si, gj):
            print(f"  sample{i} == gt{j} EXACT MATCH!")

# Check if sample0 is gt12 just visually similar
print(f"\nsample0[0,:5] = {s0[0,:5,0]}")
print(f"gt12[0,:5]  = {g12[0,:5,0]}")
print(f"gt0[0,:5]   = {g0[0,:5,0]}")
