import re
import numpy as np

log_p = "/root/Workspace/xy/DiT/assets/results/unet_dic_100k_b768/20261018-063716-Unet-flow/train.log"

steps = []
diffs = []
lrs = []
grad_norms = []

with open(log_p, encoding="utf-8") as f:
    for line in f:
        m = re.search(r"\(step=(\d+)\)\s+Diff:\s+([0-9\.]+)\s+\|\s+.*LR:\s+([0-9\.e\-]+)", line)
        if m:
            s = int(m.group(1))
            d = float(m.group(2))
            lr = float(m.group(3))
            steps.append(s)
            diffs.append(d)
            lrs.append(lr)
        gn_m = re.search(r"\(step=(\d+)\)\s+\[diag\]\s+grad-norm:\s+blocks=([0-9\.]+)\s+other=([0-9\.]+)", line)
        if gn_m:
            grad_norms.append((int(gn_m.group(1)), float(gn_m.group(2)), float(gn_m.group(3))))

print(f"Total logged steps: {len(steps)} (from step {steps[0]} to {steps[-1]})")

print("=" * 85)
print(f"{'Step Interval':16s} | {'Mean Diff Loss':15s} | {'Min Loss':10s} | {'Loss Trend':15s} | {'Current LR':10s}")
print("=" * 85)

steps = np.array(steps)
diffs = np.array(diffs)

intervals = list(range(0, steps[-1] + 5000, 5000))
prev_mean = None
for i in range(len(intervals) - 1):
    low, high = intervals[i], intervals[i+1]
    mask = (steps > low) & (steps <= high)
    if not np.any(mask): continue
    seg_diffs = diffs[mask]
    mean_val = np.mean(seg_diffs)
    min_val = np.min(seg_diffs)
    delta_str = f"{mean_val - prev_mean:+.5f}" if prev_mean is not None else "   ---   "
    cur_lr = lrs[np.where(mask)[0][-1]]
    print(f"{low:6d} -> {high:6d} | {mean_val:15.5f} | {min_val:10.5f} | {delta_str:15s} | {cur_lr:10.2e}")
    prev_mean = mean_val

print("=" * 85)
print("\n=== Periodic Gradient Norms ===")
print(f"{'Step':8s} | {'Blocks Grad Norm':18s} | {'Other Grad Norm':18s}")
print("-" * 50)
for s, b_gn, o_gn in grad_norms:
    print(f"{s:8d} | {b_gn:18.4f} | {o_gn:18.4f}")
