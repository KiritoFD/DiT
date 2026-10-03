# -*- coding: utf-8 -*-
"""验证 load_main_model 能加载 s28 ckpt（std_dino 支持 + 结构匹配）。"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, os.getcwd())
import torch, json

from src.model.controlnet import load_main_model

cfg = json.load(open("src/train/configs/s29_ctrl_gt_skel_1px.json", encoding="utf-8"))
ckpt = "assets/results/s28_std_dino_pretrain/20260831-204702-s28-std-dino-pretrain/checkpoints/0001000.pt"

print("=== load_main_model (s28 ckpt -> s29 main) ===")
model = load_main_model(
    model_name=cfg["model"], ckpt_path=ckpt, device="cpu",
    num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
    condition_fusion=cfg["condition_fusion"],
    callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
    char_proj_mode=cfg["char_proj_mode"],
    freeze_char_table=False,
    use_std_dino_char_embedder=cfg.get("use_std_dino_char_embedder", False),
    std_dino_table_path=cfg.get("std_dino_table_path"),
    cond_drop_all_prob=cfg["cond_drop_all_prob"],
    cond_drop_one_prob=cfg["cond_drop_one_prob"],
    cond_drop_which_glyph_prob=cfg["cond_drop_which_glyph_prob"],
    use_checkpoint=False, learn_sigma=False,
    norm_type=cfg["norm_type"], mlp_type=cfg["mlp_type"],
    qk_norm=bool(cfg["qk_norm"]), rope=bool(cfg["rope"]),
    rope_theta=cfg["rope_theta"], attn_impl="sdpa",
)
emb = model.y_char_embedder
print(f"char_embedder: {type(emb).__name__}  table={tuple(emb.char_table.shape)}")
# 打印参数数量
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"main model trainable params: {n:,}")
print("LOAD OK")
