import os
import json

cfgs = {
    "v10b": "/root/Workspace/xy/DiT/src/train/configs/v10b_stdskel_fame3_c41x_cos_e.json",
    "v13": "/root/Workspace/xy/DiT/src/train/configs/v13_base_50k.json",
    "v21": "/root/Workspace/xy/DiT/src/train/configs/v21_skelnet_200k.json",
    "v23": "/root/Workspace/xy/DiT/src/train/configs/v23_splitnorm.json",
    "v66": "/root/Workspace/xy/DiT/src/train/configs/v66_tables_condroute2456.json",
    "v68": "/root/Workspace/xy/DiT/src/train/configs/v68_aug_sp_c2ot.json",
    "v70": "/root/Workspace/xy/DiT/src/train/configs/v70_aug_sp_stdskel_c2ot.json"
}

print("=== Analyzing Milestone Architectures ===")
for name, path in cfgs.items():
    if not os.path.exists(path):
        print(f"[{name}] Config not found: {path}")
        continue
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    print(f"\n--- {name} ---")
    keys = [
        "model", "dit_type", "skel_as_glyph_cond", "no_char_cond", "glyph_inject_layers",
        "glyph_inject_at", "glyph_inject_mode", "glyph_embedder_depth", "use_char_cond",
        "char_embed_dim", "callig_embed_dim", "font_embed_dim", "num_characters",
        "num_calligraphers", "num_fonts", "image_size", "patch_size", "in_channels",
        "skelnet", "deform_net", "deform_scale", "split_ln", "eval_csv"
    ]
    summary = {k: c.get(k) for k in keys if k in c}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
