# -*- coding: utf-8 -*-
"""Inspect ctrl ckpt keys to debug resume missing=140."""
import sys, os, torch, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "controlnet"))
from controlnet_dit import ControlNetDiT, load_main_model

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "5script/results/ctrl_skel/20260828-103523-ctrl-skel-s18-flow/checkpoints/0005000.pt"
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("top keys:", list(ck.keys()))
for tk in ("ctrl", "ema", "model", "ema_model"):
    if tk in ck and ck[tk]:
        ks = list(ck[tk].keys())
        print(f"\n[{tk}] {len(ks)} keys; first 5:")
        for k in ks[:5]:
            print("  ", k, tuple(ck[tk][k].shape))
        # check prefix
        prefixes = set(k.split(".")[0] for k in ks)
        print(f"  prefixes: {prefixes}")
        # check ctrl_encoder sub-structure
        ce = [k for k in ks if k.startswith("ctrl_encoder")]
        print(f"  ctrl_encoder keys: {len(ce)}; first 3: {ce[:3]}")

# Build model and test load
print("\n--- building ctrl to check missing ---")
m = load_main_model(ckpt_path="5script/results/s18_s_flow_small/20260827-232003-s18-s-flow-small/checkpoints/0043000.pt", device="cpu",
    num_calligraphers=1011, num_characters=35130, condition_fusion="factorized_add",
    callig_embed_dim=128, char_embed_dim=384, char_proj_mode="ln_only", freeze_char_table=True)
ctrl = ControlNetDiT(m, cond_in_channels=1, train_ctrl_only=True)
model_keys = [k for k in ctrl.state_dict().keys() if k.startswith("ctrl_encoder")]
print(f"model ctrl_encoder keys: {len(model_keys)}; first 5:")
for k in model_keys[:5]:
    print("  ", k)

# Try loading ema
ctrl_src = ck.get("ema") or ck.get("ctrl")
ctrl_keys = {k: v for k, v in ctrl_src.items() if k.startswith("ctrl_encoder")}
print(f"\nctrl_src has {len(ctrl_src)} keys; filtered to ctrl_encoder: {len(ctrl_keys)}")
miss, unexp = ctrl.load_state_dict(ctrl_keys, strict=False)
ce_miss = [k for k in miss if k.startswith("ctrl_encoder")]
ce_unexp = [k for k in unexp if k.startswith("ctrl_encoder")]
print(f"missing ctrl_encoder: {len(ce_miss)}; first 5: {ce_miss[:5]}")
print(f"unexpected ctrl_encoder: {len(ce_unexp)}; first 5: {ce_unexp[:5]}")
# Compare names
saved_set = set(ctrl_keys.keys())
model_set = set(model_keys)
only_model = model_set - saved_set
only_saved = saved_set - model_set
print(f"\nin model not in saved: {len(only_model)}; first 5: {sorted(only_model)[:5]}")
print(f"in saved not in model: {len(only_saved)}; first 5: {sorted(only_saved)[:5]}")
