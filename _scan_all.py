import json, glob, os, re
os.chdir("/root/Workspace/xy/DiT")

def best_of(dirname):
    fs = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_*.json"))
    if not fs: return None
    best = None
    for f in fs:
        try:
            d = json.load(open(f))
        except: continue
        if best is None or d.get("ssim", -999) > best.get("ssim", -999):
            best = d
    return best

def max_step_of(dirname):
    ckpts = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/*.pt"))
    if not ckpts: return None
    m = re.search(r'(\d+)\.pt$', os.path.basename(ckpts[-1]))
    return int(m.group(1)) if m else None

def hours_of(dirname):
    ckpts = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/*.pt"))
    if len(ckpts) < 2: return None
    return (os.path.getmtime(ckpts[-1]) - os.path.getmtime(ckpts[0])) / 3600.0

# Scan ALL result dirs
dirs = sorted([d for d in os.listdir("5script/results") if os.path.isdir(f"5script/results/{d}")])
print(f"{'dir':<38} {'best_step':>9} {'SSIM':>8} {'LPIPS':>8} {'SkelIoU':>8} {'MSE':>8} {'hours':>7} {'max_step':>9}")
print("-" * 110)
for d in dirs:
    b = best_of(d)
    if b is None:
        continue
    ms = max_step_of(d)
    h = hours_of(d)
    h_str = f"{h:.1f}h" if h else "?"
    print(f"{d:<38} {b.get('step',0):>9} {b.get('ssim',-1):>8.4f} {b.get('lpips',-1):>8.4f} {b.get('skel_iou',-1):>8.4f} {b.get('mse',-1):>8.4f} {h_str:>7} {str(ms) if ms else '?':>9}")
