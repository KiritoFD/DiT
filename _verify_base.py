import json, glob, os
os.chdir("/root/Workspace/xy/DiT")

# Check ALL runs of every controlnet exp for resume_full
exps = ["s6_top6_struct_fp32", "s7_ramp_b8all", "s8_structv2_b8all", "s9_skelonly", "s6_top6_diffonly"]
for e in exps:
    cfgs = sorted(glob.glob(f"5script/results/{e}/*/resolved_config.json"))
    print(f"=== {e} ({len(cfgs)} runs) ===")
    for cf in cfgs:
        cfg = json.load(open(cf))
        rf = cfg.get("resume_full")
        run = os.path.basename(os.path.dirname(cf))
        print(f"  {run}: resume_full={rf}")
    print()

# Verify base ckpt exists
base = "5script/results/s6_top6_diffonly/20260820-191536-s6-top6-diffonly-resume/checkpoints/0195000.pt"
print(f"base 195k exists: {os.path.exists(base)}  size={os.path.getsize(base)/1e6:.0f}MB" if os.path.exists(base) else f"base 195k MISSING")
base2 = "5script/results/s6_top6_diffonly/20260820-191536-s6-top6-diffonly-resume/checkpoints/0200000.pt"
print(f"base 200k exists: {os.path.exists(base2)}" if os.path.exists(base2) else f"base 200k MISSING")

# List all s6 diffonly ckpts
ckpts = sorted(glob.glob("5script/results/s6_top6_diffonly/*/checkpoints/*.pt"))
print(f"\ns6_top6_diffonly ckpts ({len(ckpts)}):")
for c in ckpts:
    print(f"  {os.path.basename(c)}  {os.path.getsize(c)/1e6:.0f}MB")
