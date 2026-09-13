import os, glob, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base = "/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear/20260824-151758-s10-b4-grey-clear/checkpoints/eval_samples"
for d in sorted(os.listdir(base)):
    full = os.path.join(base, d)
    samples = glob.glob(os.path.join(full, "sample*.png"))
    gts = glob.glob(os.path.join(full, "gt*.png"))
    print(f"{d}: {len(samples)} samples, {len(gts)} gts")
