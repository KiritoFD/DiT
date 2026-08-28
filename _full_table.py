import json, glob, os, re
from datetime import datetime
os.chdir("/root/Workspace/xy/DiT")

def curve(dirname):
    fs = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_*.json"))
    out = []
    for f in fs:
        d = json.load(open(f))
        out.append(d)
    return out

def best(dicts, metric="ssim"):
    if not dicts: return None
    return max(dicts, key=lambda d: d.get(metric, -999))

def train_time_from_log(tag):
    log = f"/tmp/{tag}_full.log"
    if not os.path.exists(log): log = f"/tmp/{tag}_train.log"
    if not os.path.exists(log): log = f"/tmp/{tag}.log"
    if not os.path.exists(log): return None
    lines = open(log, errors="replace").readlines()
    start_ts = end_ts = None
    max_step = 0
    for line in lines:
        m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?step=(\d+)', line)
        if m:
            if start_ts is None: start_ts = m.group(1)
            end_ts = m.group(1)
            step = int(m.group(2))
            if step > max_step: max_step = step
    if start_ts and end_ts:
        t0 = datetime.strptime(start_ts, "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(end_ts, "%Y-%m-%d %H:%M:%S")
        return (t1-t0).total_seconds()/3600.0, max_step
    return None, max_step

def train_time_from_dir(dirname):
    """Estimate from dir mtime."""
    dirs = glob.glob(f"5script/results/{dirname}/*/")
    if not dirs: return None, None
    # Check ckpt timestamps
    ckpts = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/*.pt"))
    if len(ckpts) < 2: return None, None
    t0 = os.path.getmtime(ckpts[0])
    t1 = os.path.getmtime(ckpts[-1])
    # Get max step from ckpt names
    max_step = 0
    for c in ckpts:
        m = re.search(r'(\d+)\.pt', os.path.basename(c))
        if m: max_step = max(max_step, int(m.group(1)))
    return (t1-t0)/3600.0, max_step

# ── All experiments ──
experiments = [
    # (tag, dirname, model, diff_type, dataset, description)
    ("s6",  "s6_top6_diffonly",     "S/2",  "DDPM", "top6",   "top6 base (diff only)"),
    ("s9",  "s9_b4_clean_dino",     "B/2",  "DDPM", "top6",   "top6 B/2 clean dino"),
    ("s11", "s11_top6_p4",          "S/2",  "DDPM", "top6",   "top6 base S/2"),
    ("s12", "s12_3top30_dino",      "S/2",  "DDPM", "3top30", "3top30 DDPM S/2"),
    ("s13", "s13_3top30_dino_xs",   "XS/2", "DDPM", "3top30", "3top30 DDPM XS/2"),
    ("s15", "s15_ws_flow",         "WS/2", "Flow", "3top30", "3top30 Flow WS/2"),
    ("s17", "s17_s_flow",           "S/2",  "Flow", "3top30", "3top30 Flow S/2"),
    ("s14", "s14_ws_ddpm",          "WS/2", "DDPM", "3top30", "3top30 DDPM WS/2"),
    ("s16", "s16_s_ddpm",           "S/2",  "DDPM", "3top30", "3top30 DDPM S/2"),
]

# Check for controlnet experiments
for d in ["ctrl_skel", "skel_decoder", "compositional"]:
    c = curve(d)
    if c:
        experiments.append(("ctrl", d, "?", "DDPM", "top6", f"top6 controlnet ({d})"))

print("=" * 120)
print(f"{'tag':<5} {'model':<6} {'diff':<6} {'dataset':<8} {'desc':<30} {'best_step':>9} {'SSIM':>8} {'LPIPS':>8} {'SkelIoU':>8} {'MSE':>8} {'hours':>7} {'max_step':>9}")
print("-" * 120)
for tag, dirname, model, diff_type, dataset, desc in experiments:
    c = curve(dirname)
    if not c:
        print(f"{tag:<5} {model:<6} {diff_type:<6} {dataset:<8} {desc:<30} (no eval data)")
        continue
    b = best(c, "ssim")
    ssim_v = b.get('ssim', -1)
    lpips_v = b.get('lpips', -1)
    skel_v = b.get('skel_iou', b.get('skeleton_iou', b.get('skel', -1)))
    mse_v = b.get('mse', -1)
    step_v = b.get('step', -1)
    # training time
    result = train_time_from_log(tag)
    if result is None or result[0] is None:
        result2 = train_time_from_dir(dirname)
        if result2[0] is not None:
            h, ms = result2
        else:
            h, ms = None, None
    else:
        h, ms = result
    h_str = f"{h:.1f}h" if h else "?"
    ms_str = f"{ms}" if ms else f"{step_v}"
    print(f"{tag:<5} {model:<6} {diff_type:<6} {dataset:<8} {desc:<30} {step_v:>9} {ssim_v:>8.4f} {lpips_v:>8.4f} {skel_v:>8.4f} {mse_v:>8.4f} {h_str:>7} {ms_str:>9}")

print("\n" + "=" * 120)
print("\n=== FULL EVAL CURVES (key steps) ===")
for tag, dirname, model, diff_type, dataset, desc in experiments:
    c = curve(dirname)
    if not c: continue
    print(f"\n--- {tag} {model} {diff_type} {dataset} ({desc}) ---")
    for d in c:
        step = d["step"]
        if step <= 5000 or step % 50000 == 0 or step == c[-1]["step"]:
            print(f"  step={step:6d}  SSIM={d.get('ssim',-1):.4f}  LPIPS={d.get('lpips',-1):.4f}  SkelIoU={d.get('skel_iou',-1):.4f}  MSE={d.get('mse',-1):.4f}")

# List all results dirs for completeness
print("\n\n=== ALL RESULTS DIRS ===")
import subprocess
r = subprocess.run(["ls", "-d", "5script/results/*/"], capture_output=True, text=True)
print(r.stdout)
