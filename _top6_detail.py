import json, glob, os, re
os.chdir("/root/Workspace/xy/DiT")

def curve(dirname):
    fs = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/eval_auto_*.json"))
    return [json.load(open(f)) for f in fs]

def hours_of(dirname):
    ckpts = sorted(glob.glob(f"5script/results/{dirname}/*/checkpoints/*.pt"))
    if len(ckpts) < 2: return None
    return (os.path.getmtime(ckpts[-1]) - os.path.getmtime(ckpts[0])) / 3600.0

# top6 related experiments
exps = {
    "s6_top6_diffonly": "top6 base DIFF-ONLY",
    "s6_top6_struct_fp32": "top6 controlnet struct_fp32",
    "s6_top6_struct_fp32_full": "top6 controlnet struct_fp32_full",
    "s7_ramp_b8all": "top6 controlnet ramp_b8all",
    "s8_structv2_b8all": "top6 controlnet structv2_b8all",
    "s9_skelonly": "top6 controlnet skelonly",
    "s11_top6_p4": "top6 S/4 p4",
    "s10_b4_grey_clear": "top6 B/4 grey_clear",
}

for e, label in exps.items():
    c = curve(e)
    if not c:
        print(f"=== {label} ({e}): no eval ===")
        continue
    b = max(c, key=lambda d: d.get("ssim", -999))
    h = hours_of(e)
    # last few points
    tail = c[-3:]
    print(f"=== {label} ({e}) | best SSIM {b.get('ssim',-1):.4f} @step {b.get('step',0)} | {h:.1f}h if h else '?' ===")
    for d in c:
        if d["step"] % 10000 == 0 or d == c[-1]:
            print(f"    step={d['step']:6d}  SSIM={d.get('ssim',-1):.4f}  LPIPS={d.get('lpips',-1):.4f}  SkelIoU={d.get('skel_iou',-1):.4f}  MSE={d.get('mse',-1):.4f}")
    print()
