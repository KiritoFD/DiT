import json, glob, os, re
os.chdir("/root/Workspace/xy/DiT")

def get_eval_curve(dirname):
    fs = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_*.json"))
    if not fs: return []
    out = []
    for f in fs:
        d = json.load(open(f))
        out.append((d["step"], d["ssim"], d["lpips"], d["skel_iou"], d["mse"]))
    return out

def get_train_time(dirname):
    """Estimate train time from first and last eval_auto timestamps."""
    fs = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_*.json"))
    if len(fs) < 2: return None
    d0 = json.load(open(fs[0]))
    d1 = json.load(open(fs[-1]))
    # Not reliable from json. Try from log file.
    return None

def get_train_time_from_log(tag):
    """Parse train log for start/end timestamps."""
    log = f"/tmp/{tag}_full.log"
    if not os.path.exists(log): return None, None, None
    lines = open(log, errors="replace").readlines()
    start_ts = end_ts = None
    max_step = 0
    for line in lines:
        m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*step=(\d+)', line)
        if m:
            if start_ts is None: start_ts = m.group(1)
            end_ts = m.group(1)
            step = int(m.group(2))
            if step > max_step: max_step = step
    return start_ts, end_ts, max_step

def calc_hours(start, end):
    if not start or not end: return None
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    t0 = datetime.strptime(start, fmt)
    t1 = datetime.strptime(end, fmt)
    return (t1 - t0).total_seconds() / 3600.0

# ── s12 DDPM (3top30, S/2, 135k steps) ──
print("=== s12 DDPM (S/2, 3top30) ===")
s12 = get_eval_curve("s12_3top30_dino")
if s12:
    for step, ssim, lpips, skel, mse in s12:
        if step % 25000 == 0 or step == s12[-1][0] or step <= 5000:
            print(f"  step={step:6d}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  SkelIoU={skel:.4f}  MSE={mse:.4f}")

# ── s11 DDPM (top6, S/2, 195k steps) ──
print("\n=== s11 DDPM (S/2, top6) ===")
s11 = get_eval_curve("s11_top6_p4")
if s11:
    for step, ssim, lpips, skel, mse in s11:
        if step % 25000 == 0 or step == s11[-1][0] or step <= 5000:
            print(f"  step={step:6d}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  SkelIoU={skel:.4f}  MSE={mse:.4f}")

# ── s13 DDPM (XS/2, 3top30, ~2.5k steps, killed early) ──
print("\n=== s13 DDPM (XS/2, 3top30) ===")
s13 = get_eval_curve("s13_3top30_dino_xs")
if s13:
    for step, ssim, lpips, skel, mse in s13:
        print(f"  step={step:6d}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  SkelIoU={skel:.4f}  MSE={mse:.4f}")

# ── Current Flow runs ──
print("\n=== s15 Flow (WS/2, 3top30) — COMPLETED ===")
s15 = get_eval_curve("s15_ws_flow")
if s15:
    for step, ssim, lpips, skel, mse in s15:
        if step % 25000 == 0 or step == s15[-1][0] or step <= 5000:
            print(f"  step={step:6d}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  SkelIoU={skel:.4f}  MSE={mse:.4f}")

print("\n=== s17 Flow (S/2, 3top30) — RUNNING ===")
s17 = get_eval_curve("s17_s_flow")
if s17:
    for step, ssim, lpips, skel, mse in s17:
        if step % 25000 == 0 or step == s17[-1][0] or step <= 5000:
            print(f"  step={step:6d}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  SkelIoU={skel:.4f}  MSE={mse:.4f}")

# ── Training times ──
print("\n=== TRAINING TIMES ===")
for tag in ["s15", "s17"]:
    s, e, ms = get_train_time_from_log(tag)
    h = calc_hours(s, e)
    if h: print(f"  {tag}: {h:.1f}h, max_step={ms}")

# s12 from log
for tag in ["s12"]:
    s, e, ms = get_train_time_from_log(tag)
    h = calc_hours(s, e)
    if h: print(f"  {tag}: {h:.1f}h, max_step={ms}")

# Check s12 from dir timestamp
import os.path
dirs12 = glob.glob("5script/results/s12_3top30_dino/*")
if dirs12:
    dn = os.path.basename(dirs12[0])
    print(f"  s12 dir: {dn}")
