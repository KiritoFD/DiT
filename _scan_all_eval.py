import json, glob, os
os.chdir("/root/Workspace/xy/DiT")

# Scan ALL eval_auto_*.json across ALL result dirs
fs = sorted(glob.glob("5script/results/*/checkpoint*/eval_auto_*.json") +
            glob.glob("5script/results/*/*/checkpoints/eval_auto_*.json"))

print(f"Total eval files: {len(fs)}")
print()

# Find anything with SSIM > 0.6
high = []
for f in fs:
    try:
        d = json.load(open(f))
        ssim = d.get("ssim", -1)
        if ssim > 0.5:
            parts = f.split("/")
            exp = parts[2] if len(parts) > 2 else "?"
            high.append((ssim, d.get("step", 0), exp, f, d))
    except:
        pass

high.sort(key=lambda x: -x[0])
print(f"=== ALL evals with SSIM > 0.5 (sorted by SSIM desc) ===")
print(f"{'SSIM':>8} {'step':>9} {'exp':<38} {'LPIPS':>8} {'SkelIoU':>8} {'MSE':>8}")
for ssim, step, exp, f, d in high[:40]:
    print(f"{ssim:>8.4f} {step:>9} {exp:<38} {d.get('lpips',-1):>8.4f} {d.get('skel_iou',-1):>8.4f} {d.get('mse',-1):>8.4f}")

# Also check eval_samples for any eval.json (not eval_auto) - maybe older format
fs2 = sorted(glob.glob("5script/results/*/checkpoint*/eval_*.json") +
             glob.glob("5script/results/*/*/checkpoints/eval_*.json"))
# filter out eval_auto and eval_pending
fs2 = [f for f in fs2 if "eval_auto" not in f and "eval_pending" not in f and "eval_samples" not in f]
print(f"\n=== Non-auto eval files: {len(fs2)} ===")
for f in fs2[:20]:
    try:
        d = json.load(open(f))
        ssim = d.get("ssim", d.get("SSIM", -1))
        step = d.get("step", "?")
        exp = f.split("/")[2]
        print(f"  {f}  SSIM={ssim}  step={step}")
    except:
        print(f"  {f} (parse error)")
