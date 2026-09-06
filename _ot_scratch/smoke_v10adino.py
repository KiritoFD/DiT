# -*- coding: utf-8 -*-
"""v10adino CPU 冒烟: 构建 v10a-dino 模型 (冻结 DINO char 表) + fwd/CFG/bwd。

验证点:
1. StdDinoCharEmbedder 正确加载 384 PCA 表, 冻结 buffer (不可训练), 仅 null_embed 可学
2. char_proj_mode=ln_only 直通 (LayerNorm, 无 Linear)
3. 参数量: v10a 46.5M -> v10a-dino 应约 -13.8M (char 表 35130*384 换 7026*384 冻结 buffer)
   但注意: LabelEmbedder 35130x384 参数被 DINO 7026x384 buffer(非参数) 替换 + char_proj mlp->ln_only
4. forward 各条件组合 + forward_with_cfg + backward
5. skel_as_glyph_cond 通路 (g 注入) 与新 embedder 共存
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, os.getcwd())
import torch
import json

from src.model import DiT_2Cond_models

cfg = json.load(open("src/train/configs/v10adino_skel_cond_pretrain.json", encoding="utf-8"))
print("=== build model (CPU) ===")
model = DiT_2Cond_models[cfg["model"]](
    input_size=32, in_channels=4, learn_sigma=False,
    num_calligraphers=cfg["num_calligraphers"], num_characters=cfg["num_characters"],
    condition_fusion=cfg["condition_fusion"],
    callig_embed_dim=cfg["callig_embed_dim"], char_embed_dim=cfg["char_embed_dim"],
    char_proj_mode=cfg.get("char_proj_mode", "full"),
    cond_drop_all_prob=cfg["cond_drop_all_prob"], cond_drop_one_prob=cfg["cond_drop_one_prob"],
    cond_drop_which_glyph_prob=cfg["cond_drop_which_glyph_prob"],
    use_glyph_cond=True, use_char_cond=True,
    glyph_scale_init=cfg["glyph_scale_init"], glyph_drop_prob=cfg["glyph_drop_prob"],
    use_std_dino_char_embedder=cfg.get("use_std_dino_char_embedder", False),
    std_dino_table_path=cfg.get("std_dino_table_path"),
    freeze_char_table=cfg.get("freeze_char_table", False),
    norm_type=cfg.get("norm_type","rms"), mlp_type=cfg.get("mlp_type","swiglu"),
    qk_norm=bool(cfg.get("qk_norm",1)), rope=bool(cfg.get("rope",1)),
    rope_theta=cfg.get("rope_theta",100.0), attn_impl="eager",  # CPU base: xformers mem_efficient 仅 cuda
)
model.eval()

emb = model.y_char_embedder
print("char_embedder:", type(emb).__name__)
print("  table:", tuple(emb.char_table.shape), "| downsample:", emb.downsample)
print("  table requires_grad:", emb.char_table.requires_grad, "(buffer, 应 False)")
emb_train = sum(p.numel() for p in emb.parameters() if p.requires_grad)
print("  embedder trainable params:", emb_train, "(应=384 null_embed)")
print("  char_proj:", type(model.char_proj).__name__,
      "| weight shape:", tuple(model.char_proj.weight.shape) if hasattr(model.char_proj, 'weight') else None)
total_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("  TOTAL trainable params:", f"{total_train:,}")

# 冻结验证: DINO 表值 = 加载表值, 永不被梯度改
import numpy as np
raw = np.load(cfg["std_dino_table_path"]).astype(np.float32)
same = np.allclose(emb.char_table.numpy(), raw, atol=1e-6)
print("  char_table == source file:", same)

# forward: 训练态 (char dropout 生效)
B = 4
x = torch.randn(B, 4, 32, 32)
t = torch.rand(B)
y_callig = torch.randint(0, cfg["num_calligraphers"], (B,)).long()
y_char = torch.randint(0, cfg["num_characters"], (B,)).long()
g = torch.randn(B, 4, 32, 32)
model.train()
out = model(x, t, y_callig, y_char, g=g)
print("train fwd out:", tuple(out.shape))

# forward_with_cfg
with torch.no_grad():
    out_cfg = model.forward_with_cfg(x, t, y_callig, y_char, cfg_scale=0.7, g=g)
print("cfg fwd out:", tuple(out_cfg.shape))

# backward
loss = out.float().pow(2).mean() + out_cfg.float().pow(2).mean()
loss.backward()
grad_ok = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
print("backward grads:", grad_ok, "/", sum(1 for p in model.parameters() if p.requires_grad))

# REPA 模块构建 (w_repa>0 时 train.py 会挂 RepaModule —— 这里只验证模型侧被请求中间层)
try:
    inter = model(x, t, y_callig, y_char, g=g, return_intermediate_layers=(8, 11))
    if isinstance(inter, tuple):
        feats = inter[1]
        print("intermediate layers:", {k: tuple(v.shape) for k, v in feats.items()} if isinstance(feats, dict) else [tuple(v.shape) for v in feats])
except Exception as e:
    print("intermediate check fail:", type(e).__name__, e)

print("SMOKE_DONE")