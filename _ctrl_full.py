import json, glob, os
os.chdir("/root/Workspace/xy/DiT")

# Full ctrl_skel eval table
fs = sorted(glob.glob("5script/results/ctrl_skel/*/checkpoints/eval_auto_*.json"),
            key=lambda f: int(os.path.basename(f).replace("eval_auto_","").replace(".json","")))
print(f"=== ctrl_skel (warm-start top6) eval: {len(fs)} steps ===")
print(f"{'step':>7} {'MSE_base':>10} {'SSIM_base':>10} {'MSE_ctrl':>10} {'SSIM_ctrl':>10} {'dMSE':>9} {'dSSIM':>9}")
print("-"*75)
for f in fs:
    d = json.load(open(f))
    step = d.get("step", 0)
    print(f"{step:>7} {d.get('mse_base',-1):>10.4f} {d.get('ssim_base',-1):>10.4f} "
          f"{d.get('mse_ctrl',-1):>10.4f} {d.get('ssim_ctrl',-1):>10.4f} "
          f"{d.get('delta_mse',-1):>9.4f} {d.get('delta_ssim',-1):>9.4f}")

# Also check compositional
fs2 = sorted(glob.glob("5script/results/compositional/*/checkpoints/eval_auto_*.json"),
            key=lambda f: int(os.path.basename(f).replace("eval_auto_","").replace(".json","")))
if fs2:
    print(f"\n=== compositional eval: {len(fs2)} steps ===")
    d0 = json.load(open(fs2[0]))
    print(f"fields: {list(d0.keys())}")
    for f in fs2:
        d = json.load(open(f))
        step = d.get("step", 0)
        print(f"  step={step}  {dict((k,round(v,4)) for k,v in d.items() if k!='step')}")

# Check skel_decoder too
for exp in ["skel_decoder", "skel_decoder_d3"]:
    fs3 = sorted(glob.glob(f"5script/results/{exp}/*/checkpoints/eval_auto_*.json"),
                key=lambda f: int(os.path.basename(f).replace("eval_auto_","").replace(".json","")))
    if fs3:
        print(f"\n=== {exp}: {len(fs3)} evals ===")
        d0 = json.load(open(fs3[0]))
        print(f"fields: {list(d0.keys())}")
        for f in fs3[:5]:
            d = json.load(open(f))
            print(f"  {dict((k,round(v,4) if isinstance(v,float) else v) for k,v in d.items())}")
