import json, glob, os, re
from datetime import datetime
os.chdir("/root/Workspace/xy/DiT")

def curve(dirname):
    fs = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_*.json"))
    return [json.load(open(f)) for f in fs]

def best(dicts, metric="ssim"):
    return max(dicts, key=lambda d: d.get(metric, -999)) if dicts else None

def train_hours_from_ckpt(dirname):
    ckpts = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/*.pt"))
    if len(ckpts) < 2: return None
    t0 = os.path.getmtime(ckpts[0])
    t1 = os.path.getmtime(ckpts[-1])
    return (t1 - t0) / 3600.0

def max_step_from_ckpt(dirname):
    ckpts = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/*.pt"))
    if not ckpts: return None
    m = re.search(r'(\d+)\.pt$', os.path.basename(ckpts[-1]))
    return int(m.group(1)) if m else None

# All experiments, grouped
groups = {
    "top6 base": [
        ("s6", "s6_top6_diffonly", "S/2", "DDPM", "top6 base (diff only)"),
        ("s11", "s11_top6_p4", "S/2", "DDPM", "top6 base S/2 (dino+struct)"),
    ],
    "top6 controlnet": [
        ("s6s", "s6_top6_struct_fp32", "S/2", "DDPM", "top6 controlnet struct fp32"),
        ("s6sf", "s6_top6_struct_fp32_full", "S/2", "DDPM", "top6 controlnet struct full"),
        ("s8v2", "s8_structv2_b8all", "B/2", "DDPM", "top6 controlnet structv2 b8all"),
        ("s8v2s", "s8_structv2_b32sub8", "B/2", "DDPM", "top6 controlnet structv2 b32sub8"),
        ("s5c", "s5_2factor_B_canny05_pixelsk", "B/2", "DDPM", "top6 controlnet canny+pixelsk"),
        ("s5ls", "s5_2factor_B_latentstruct", "B/2", "DDPM", "top6 controlnet latentstruct"),
        ("s9sk", "s9_skelonly", "B/2", "DDPM", "top6 controlnet skelonly"),
    ],
    "3top30": [
        ("s12", "s12_3top30_dino", "S/2", "DDPM", "3top30 DDPM S/2"),
        ("s15", "s15_ws_flow", "WS/2", "Flow", "3top30 Flow WS/2"),
        ("s17", "s17_s_flow", "S/2", "Flow", "3top30 Flow S/2"),
    ],
}

print("=" * 130)
print(f"{'tag':<6} {'model':<6} {'diff':<6} {'desc':<38} {'best_step':>9} {'SSIM':>8} {'LPIPS':>8} {'SkelIoU':>8} {'MSE':>8} {'hours':>7} {'max_step':>9}")
print("-" * 130)
for group, exps in groups.items():
    print(f"\n### {group} ###")
    for tag, dirname, model, diff, desc in exps:
        c = curve(dirname)
        if not c:
            print(f"{tag:<6} {model:<6} {diff:<6} {desc:<38} (no eval data)")
            continue
        b = best(c, "ssim")
        h = train_hours_from_ckpt(dirname)
        ms = max_step_from_ckpt(dirname)
        h_str = f"{h:.1f}h" if h else "?"
        ms_str = str(ms) if ms else "?"
        lpips = b.get('lpips', -1)
        skel = b.get('skel_iou', b.get('skeleton_iou', b.get('skel', -1)))
        print(f"{tag:<6} {model:<6} {diff:<6} {desc:<38} {b.get('step',0):>9} {b.get('ssim',-1):>8.4f} {lpips:>8.4f} {skel:>8.4f} {b.get('mse',-1):>8.4f} {h_str:>7} {ms_str:>9}")

print("\n" + "=" * 130)
