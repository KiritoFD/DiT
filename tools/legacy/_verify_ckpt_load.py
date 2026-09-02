# -*- coding: utf-8 -*-
"""验证新增参数不破坏已有 ckpt 加载（callig_scale/char_scale/glyph_drop）。"""
import sys, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from src.model import DiT_2Cond_models

m = DiT_2Cond_models["DiT-2Cond-S/2"](
    input_size=32, in_channels=4, num_calligraphers=1013, num_characters=35130,
    use_checkpoint=False, learn_sigma=False, condition_fusion="factorized_add",
    callig_embed_dim=128, char_embed_dim=384, cond_drop_all_prob=0.05,
    cond_drop_one_prob=0.25, skel_head_enabled=False, use_glyph_cond=False,
    glyph_scale_init=0.4, glyph_inject_layers=0, glyph_drop_prob=0.0,
    char_proj_mode="mlp", freeze_char_table=True, norm_type="rms",
    mlp_type="swiglu", qk_norm=True, rope=True, rope_theta=100.0,
    attn_impl="sdpa")

sd = torch.load("5script/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0027500.pt",
                map_location="cpu")
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
elif isinstance(sd, dict) and "ema" in sd:
    sd = sd["ema"]
sd = {k.replace("module.", ""): v for k, v in sd.items()}
miss, unexp = m.load_state_dict(sd, strict=False)
print(f"missing={len(miss)} unexpected={len(unexp)}")
print("missing sample:", miss[:6])
print("unexpected sample:", unexp[:6])
