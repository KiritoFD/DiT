import json, glob, os
os.chdir("/root/Workspace/xy/DiT")
exps = ["s6_top6_struct_fp32", "s6_top6_struct_fp32_full", "s7_ramp_b8all",
        "s8_structv2_b8all", "s9_skelonly", "s6_top6_diffonly"]
for e in exps:
    cfgs = sorted(glob.glob(f"5script/results/{e}/*/resolved_config.json"))
    if not cfgs:
        print(f"=== {e}: no config ==="); continue
    cfg = json.load(open(cfgs[-1]))
    print(f"=== {e} ===")
    for k in ["model", "pretrained", "use_canny", "use_skel", "w_canny", "w_skel",
              "w_skel_head", "global_batch_size", "char_embed_dim", "max_steps",
              "reset_cond_head", "train_cond_head", "cond_mode", "use_lora"]:
        if k in cfg and cfg[k] not in [None, False, 0, 0.0, ""]:
            print(f"  {k}: {cfg[k]}")
    print()
