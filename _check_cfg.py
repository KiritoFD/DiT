import json, glob, os
os.chdir("/root/Workspace/xy/DiT")
exps = ["s6_top6_diffonly", "s6_top6_struct_fp32", "s11_top6_p4", "s8_structv2_b8all", "s9_skelonly", "s7_ramp_b8all", "s10_b4_grey_clear"]
for e in exps:
    cfgs = sorted(glob.glob(f"5script/results/{e}/*/resolved_config.json"))
    if not cfgs:
        print(f"=== {e}: no config ===")
        continue
    cfg = json.load(open(cfgs[-1]))
    keys = ["model", "use_canny", "use_skel", "w_canny", "w_skel", "w_skel_head", "use_lora", "cond_mode", "condition_fusion", "diffusion_type", "global_batch_size", "char_embed_dim", "char_proj_mode", "freeze_char_table", "max_steps"]
    print(f"=== {e} ===")
    for k in keys:
        if k in cfg:
            print(f"  {k}: {cfg[k]}")
    print()
