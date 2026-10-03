# -*- coding: utf-8 -*-
"""stageB dry-run: 用真实 S30 ckpt 验证 train_controlnet 的加载与 ctrl 构建 (CPU 轻量, 不训练)."""
import sys
sys.path.insert(0, "/root/Workspace/xy/DiT")
import torch
from src.model.controlnet import load_main_model, ControlNetDiT

ckpt = sys.argv[1]
main = load_main_model(
    model_name="DiT-2Cond-S/2", ckpt_path=ckpt, device="cpu",
    num_calligraphers=1013, num_characters=35130, condition_fusion="factorized_add",
    callig_embed_dim=128, char_embed_dim=384, char_proj_mode="mlp", freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.30, cond_drop_which_glyph_prob=0.85,
    use_checkpoint=False, learn_sigma=None, diffusion_type="flow",
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True, rope_theta=100.0,
    attn_impl="sdpa",
)
main.eval()
print("main loaded OK, param count = %.1f M" % (sum(p.numel() for p in main.parameters()) / 1e6))

ctrl = ControlNetDiT(
    main, cond_in_channels=4, train_ctrl_only=True,
    ctrl_depth=0, ctrl_hidden=0, ctrl_num_heads=0, injection="modulate", null_cond="gaussian",
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True, rope_theta=100.0,
    attn_impl="sdpa",
)
tr = [p for p in ctrl.parameters() if p.requires_grad]
print("ctrl built OK, trainable = %.1f M" % (sum(p.numel() for p in tr) / 1e6))
print("main frozen:", all(not p.requires_grad for p in main.parameters()))

# 前向 smoke (B=2, 32x32 latent) 确保注入维度匹配
torch.manual_seed(0)
B = 2
x = torch.randn(B, 4, 32, 32)
skel = torch.randn(B, 4, 32, 32) * 0.1
y_callig = torch.randint(0, 1013, (B,))
y_char = torch.randint(0, 35130, (B,))
with torch.no_grad():
    out = ctrl(x, t=torch.rand(B), y_callig=y_callig, y_char=y_char, cond=skel, eval_only=True)
print("forward out shape:", tuple(out.shape))
print("STAGEB DRYRUN PASS")