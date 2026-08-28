import json, glob, os
os.chdir("/root/Workspace/xy/DiT")
# Dump FULL configs for the controlnet experiments to see pretrained path
exps = ["s6_top6_struct_fp32", "s7_ramp_b8all", "s8_structv2_b8all", "s9_skelonly"]
for e in exps:
    cfgs = sorted(glob.glob(f"5script/results/{e}/*/resolved_config.json"))
    if not cfgs:
        print(f"=== {e}: no config ==="); continue
    cfg = json.load(open(cfgs[-1]))
    print(f"=== {e} (full config) ===")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print()
