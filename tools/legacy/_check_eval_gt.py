import os, sys, json, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear/20260824-140208-s10-b4-grey-clear/checkpoints/eval_samples/step0001000"

# Check all GT images for grey ratio
files = sorted([f for f in os.listdir(DIR) if f.startswith("gt") and f.endswith(".png")])
print(f"GT images: {len(files)}")

results = []
for f in files:
    img = Image.open(os.path.join(DIR, f)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    grey = float(((arr > 0.15) & (arr < 0.85)).mean())
    results.append((f, grey))

results.sort(key=lambda x: -x[1])
print("\ntop 10 dirty GTs:")
for f, g in results[:10]:
    print(f"  {f}: grey={g:.3f}")

dirty = [r for r in results if r[1] > 0.2]
print(f"\ntotal GT with grey>0.2: {len(dirty)}")
dirty3 = [r for r in results if r[1] > 0.3]
print(f"total GT with grey>0.3: {len(dirty3)}")

# Also check some pred images
pred_files = sorted([f for f in os.listdir(DIR) if f.startswith("sample") and f.endswith(".png")])
print(f"\npred images: {len(pred_files)}")
pred_results = []
for f in pred_files:
    img = Image.open(os.path.join(DIR, f)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    grey = float(((arr > 0.15) & (arr < 0.85)).mean())
    pred_results.append((f, grey))
pred_results.sort(key=lambda x: -x[1])
print("top 10 dirty preds:")
for f, g in pred_results[:10]:
    print(f"  {f}: grey={g:.3f}")
