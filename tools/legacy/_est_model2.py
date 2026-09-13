# -*- coding: utf-8 -*-
"""估算 B 模型 + 实际 batch 前向测显存。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models import DiT_2Cond_models

print("=== 参数量对比 ===")
for name in ["DiT-2Cond-S/4", "DiT-2Cond-S/2", "DiT-2Cond-B/2"]:
    try:
        model = DiT_2Cond_models[name](
            input_size=64, num_calligraphers=1011, num_characters=35130,
            use_checkpoint=False, condition_fusion="factorized_add",
            callig_embed_dim=128, char_embed_dim=768,
            cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, in_channels=3)
        n = sum(p.numel() for p in model.parameters())
        nt = model.x_embedder.num_patches
        hs = model.x_embedder.proj.out_channels
        print(f"  {name}: {n/1e6:.1f}M params, {nt} tokens, hidden={hs}")
        del model
    except Exception as e:
        print(f"  {name}: ERR {e}")

# B/4 需要手动建 (没有注册)
print("\n=== DiT-2Cond-B/4 (手动) ===")
from models import DiT_2Cond
m = DiT_2Cond(input_size=64, patch_size=4, in_channels=3, hidden_size=768,
              depth=12, num_heads=12, num_calligraphers=1011, num_characters=35130,
              condition_fusion="factorized_add", callig_embed_dim=128, char_embed_dim=768,
              cond_drop_all_prob=0.05, cond_drop_one_prob=0.25)
n = sum(p.numel() for p in m.parameters())
nt = m.x_embedder.num_patches
hs = m.x_embedder.proj.out_channels
print(f"  B/4: {n/1e6:.1f}M params, {nt} tokens, hidden={hs}")
