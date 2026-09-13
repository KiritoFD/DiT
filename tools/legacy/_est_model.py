# -*- coding: utf-8 -*-
"""估算各模型配置的参数量 / token 数 / 显存需求 (latent 64x64, kl-f4).
DiT-2Cond-S/4 (current):  hidden=384, depth=12, patch=4, heads=6
DiT-2Cond-S/2:            hidden=384, depth=12, patch=2, heads=6
DiT-2Cond-B/2:            hidden=768, depth=12, patch=2, heads=12
DiT-2Cond-B/4:            hidden=768, depth=12, patch=4, heads=12
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models import DiT_2Cond_models

configs = [
    ("DiT-2Cond-S/4", 768, "current s8"),
    ("DiT-2Cond-S/2", 768, "patch=2 test"),
]

for name, char_dim, note in configs:
    model = DiT_2Cond_models[name](
        input_size=64,          # latent 64x64 (256/4)
        num_calligraphers=1011,
        num_characters=35130,
        use_checkpoint=False,
        condition_fusion="factorized_add",
        callig_embed_dim=128,
        char_embed_dim=char_dim,
        cond_drop_all_prob=0.05,
        cond_drop_one_prob=0.25,
        in_channels=3,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_patches = model.x_embedder.num_patches
    pos_shape = tuple(model.pos_embed.shape)
    print(f"\n=== {name} ({note}) ===")
    print(f"  hidden_size = {model.x_embedder.proj.out_channels}")
    print(f"  num_patches (tokens) = {n_patches}  pos_embed={pos_shape}")
    print(f"  total params = {n_params:,} ({n_params/1e6:.1f}M)")
    # char table
    ct = model.y_char_embedder.embedding_table.weight
    print(f"  char table = {tuple(ct.shape)} = {ct.numel():,} ({ct.numel()/1e6:.1f}M)")
    # char_proj
    cp = model.char_proj
    print(f"  char_proj = LayerNorm({cp[0].normalized_shape[0]}) -> Linear({cp[1].in_features}->{cp[1].out_features})")

# 估算显存: token数 x hidden^2 (attention) 是主要因素
print("\n=== 显存估算 (batch=224, latent 64x64) ===")
for name, n_tok, hs, note in [
    ("S/4", 16*16, 384, "current"),
    ("S/2", 32*32, 384, "patch=2"),
    ("B/4", 16*16, 768, "B/4"),
    ("B/2", 32*32, 768, "B/2"),
]:
    attn_mem = 224 * n_tok * hs * 4 * 2 / 1e9  # qkv 粗估
    print(f"  {name} ({note}): tokens={n_tok}, hidden={hs}, ~attn_mem={attn_mem:.1f}GB (粗估)")
