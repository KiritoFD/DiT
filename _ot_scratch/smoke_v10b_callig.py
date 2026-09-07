# -*- coding: utf-8 -*-
"""smoke_v10b_callig.py — 快速冒烟: callig 增强分支 (mlp proj + scale 1.5) 构建/fwd/CFG/bwd"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, os.getcwd())
import torch
from src.model import DiT_2Cond_models

cfg = json.load(open("src/train/configs/v10bcallig_strong.json", encoding="utf-8"))
model = DiT_2Cond_models[cfg["model"]](
    input_size=32, in_channels=4, learn_sigma=False,
    num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
    condition_fusion=cfg["condition_fusion"],
    callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
    char_proj_mode=cfg.get("char_proj_mode", "mlp"),
    callig_proj_mode=cfg.get("callig_proj_mode", "linear"),
    callig_scale_init=float(cfg.get("callig_scale_init", 1.0)),
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.3,
    cond_drop_which_glyph_prob=0.85,
    use_glyph_cond=True, use_char_cond=False,
    glyph_scale_init=cfg["glyph_scale_init"], glyph_drop_prob=cfg["glyph_drop_prob"],
    glyph_inject_layers=cfg.get("glyph_inject_layers", 0),
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
    rope_theta=100.0, attn_impl="eager")
print("callig_proj:", type(model.callig_proj).__name__, "| callig_scale:", model.callig_scale.item())
print("trainable params:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

B = 4
x = torch.randn(B, 4, 32, 32); t = torch.rand(B)
yc = torch.randint(0, cfg["num_calligraphers"], (B,)).long()
yh = torch.randint(0, cfg["num_characters"], (B,)).long()
g = torch.randn(B, 4, 32, 32)
model.train()
out = model(x, t, yc, yh, g=g)
with torch.no_grad():
    outc = model.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, g=g)
loss = out.float().pow(2).mean() + outc.float().pow(2).mean()
loss.backward()
grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
print("fwd:", tuple(out.shape), "cfg:", tuple(outc.shape), "grads:", grads)
print("SMOKE_DONE")