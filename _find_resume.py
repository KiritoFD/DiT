import json, glob, os
os.chdir("/root/Workspace/xy/DiT")
exps = ["s6_top6_struct_fp32", "s7_ramp_b8all", "s8_structv2_b8all", "s9_skelonly"]
for e in exps:
    cfgs = sorted(glob.glob(f"5script/results/{e}/*/resolved_config.json"))
    for cf in cfgs:
        cfg = json.load(open(cf))
        rf = cfg.get("resume_full")
        pt = cfg.get("pretrained")
        if rf or pt:
            print(f"{e} ({os.path.basename(os.path.dirname(cf))}): resume_full={rf}  pretrained={pt}")
