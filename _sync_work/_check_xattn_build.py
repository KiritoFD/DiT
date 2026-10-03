# -*- coding: utf-8 -*-
"""_check_xattn_build.py — 验证 xattn 注入路径能正常构建 (CPU, 不占 GPU).

背景: 52 号文档记录了"删除 xattn 与风格 token", 但代码仍在 (dit.py:721)。
      本次回到 v10b 0.5680 配方 (glyph_inject_mode=xattn, 全 12 层), 必须先确认能建起来,
      否则训练起跑才报错会浪费时间。
"""
import json
import os
import sys

import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cfg = json.load(open("src/train/configs/v11_pretrain_Sp2_base_sym.json", encoding="utf-8"))
print(f"[cfg] model={cfg['model']} inject={cfg['glyph_inject_mode']} "
      f"layers={cfg['glyph_inject_layers']} rope={cfg['rope']}")

from src.model import DiT_2Cond_models  # noqa: E402

arch = dict(norm_type=cfg.get("norm_type", "rms"),
            mlp_type=cfg.get("mlp_type", "swiglu"),
            qk_norm=bool(cfg.get("qk_norm", 1)),
            rope=bool(cfg.get("rope", 1)),
            rope_theta=float(cfg.get("rope_theta", 100.0)),
            attn_impl=cfg.get("attn_impl", "sdpa"))

model = DiT_2Cond_models[cfg["model"]](
    input_size=32,
    in_channels=4,                      # 4ch (无 aux)
    num_calligraphers=int(cfg.get("num_calligraphers", 52)),
    num_characters=int(cfg.get("num_characters") or 35130),
    condition_fusion=cfg.get("condition_fusion") or "factorized_add",
    callig_embed_dim=int(cfg.get("callig_embed_dim") or 128),
    char_embed_dim=int(cfg.get("char_embed_dim") or 384),
    char_proj_mode=cfg.get("char_proj_mode") or "mlp",
    freeze_char_table=bool(cfg.get("freeze_char_table", False)),
    cond_drop_all_prob=float(cfg.get("cond_drop_all_prob", 0.05)),
    cond_drop_one_prob=float(cfg.get("cond_drop_one_prob", 0.05)),
    cond_drop_which_glyph_prob=float(cfg.get("cond_drop_which_glyph_prob", 0.85)),
    use_checkpoint=False,
    learn_sigma=False,
    use_glyph_cond=bool(cfg.get("skel_as_glyph_cond", True)),
    use_char_cond=not bool(cfg.get("no_char_cond", False)),
    use_std_dino_char_embedder=False,
    std_dino_table_path=None,
    glyph_scale_init=float(cfg.get("glyph_scale_init", 0.6)),
    glyph_drop_prob=float(cfg.get("glyph_drop_prob", 0.1)),
    glyph_embedder_depth=int(cfg.get("glyph_embedder_depth", 2)),
    glyph_inject_layers=int(cfg.get("glyph_inject_layers", 12)),
    callig_style_attn=False,
    callig_n_style=8,
    style_token_n=0,
    style_role_init=0.02,
    glyph_inject_mode=cfg.get("glyph_inject_mode", "adaln"),
    **arch)

if cfg.get("freeze_callig_table"):
    model.y_callig_embedder.freeze_table()

n = sum(p.numel() for p in model.parameters())
print(f"[build] OK  params={n/1e6:.1f}M")
print(f"  in_channels   = {model.in_channels}")
print(f"  inject_at     = {list(model.glyph_inject_at)}")
print(f"  injections    = {type(model.glyph_injections[0]).__name__}")
print(f"  x_embedder w  = {tuple(model.x_embedder.proj.weight.shape)}  (应为 (512, 4, 2, 2))")
assert model.in_channels == 4, "应该是 4ch"
assert "Cross" in type(model.glyph_injections[0]).__name__, "xattn 模式应构建 CrossAttention 注入"
print("\n✓ xattn 路径构建正常")
