# -*- coding: utf-8 -*-
"""CPU 冒烟 v2: 修复 key 前缀 + student_dim 检测后重跑验证. 用法: python _debug_repa_smoke.py <base_ckpt>"""
import os
import sys
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _r not in sys.path:
    sys.path.insert(0, _r)
import torch
from types import SimpleNamespace

ckpt = sys.argv[1]
torch.manual_seed(0)
device = "cpu"

from src.model.controlnet import load_main_model, ControlNetDiT
from src.loss import create_diffusion_or_flow, flow_kwargs_from, REPALoss

args = SimpleNamespace(
    model="DiT-2Cond-S/2", num_calligraphers=1013, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128, char_embed_dim=384,
    char_proj_mode="mlp", freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.30, cond_drop_which_glyph_prob=0.85,
    use_checkpoint=False, diffusion_type="flow",
    norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
    rope_theta=100.0, attn_impl="sdpa",
    ctrl_depth=0, ctrl_hidden=0, ctrl_num_heads=0, injection="modulate", null_cond="gaussian",
    t_sampler="logit_normal", t_mean=0.0, t_std=1.0,
    flow_sampler="heun", heun_batch=1, shift=1.0,
)
_arch = dict(norm_type=args.norm_type, mlp_type=args.mlp_type,
             qk_norm=bool(args.qk_norm), rope=bool(args.rope),
             rope_theta=args.rope_theta, attn_impl=args.attn_impl)
_ctrl_cfg = dict(ctrl_depth=(args.ctrl_depth or None), ctrl_hidden=(args.ctrl_hidden or None),
                 ctrl_num_heads=(args.ctrl_num_heads or None),
                 injection=args.injection, null_cond=args.null_cond)

print("[1] load main ...")
main = load_main_model(model_name=args.model, ckpt_path=ckpt, device=device,
                       num_calligraphers=args.num_calligraphers,
                       num_characters=args.num_characters,
                       condition_fusion=args.condition_fusion,
                       callig_embed_dim=args.callig_embed_dim,
                       char_embed_dim=args.char_embed_dim,
                       char_proj_mode=args.char_proj_mode,
                       freeze_char_table=args.freeze_char_table,
                       cond_drop_all_prob=args.cond_drop_all_prob,
                       cond_drop_one_prob=args.cond_drop_one_prob,
                       cond_drop_which_glyph_prob=args.cond_drop_which_glyph_prob,
                       use_checkpoint=args.use_checkpoint, learn_sigma=None,
                       diffusion_type=args.diffusion_type, **_arch)
main.train()
print("[2] ControlNetDiT ...")
model = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=False,
                      **_ctrl_cfg, **_arch).to(device)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"    trainable={sum(p.numel() for p in trainable):,}")

# student_dim: S/2 latent patch 特征维 = out_channel of patch embed conv
_student_dim = main.x_embedder.proj.out_channels
print("[3] REPALoss ...")
repa = REPALoss(student_dim=_student_dim, teacher_backbone="dinov2_vits14",
                teacher_ckpt=None).to(device)
print(f"    student_dim={_student_dim}, teacher={type(repa.teacher.model).__name__}")

print("[4] flow diffusion ...")
diffusion = create_diffusion_or_flow("", diffusion_type="flow", **flow_kwargs_from(args))
print("   ", diffusion.describe())

print("[5] 前向+反向 (B=4, REPA layer=8) ...")
B = 4
torch.manual_seed(1)
x = torch.randn(B, 4, 32, 32)
y_callig = torch.randint(0, 1013, (B,))
y_char = torch.randint(0, 35130, (B,))
t = diffusion.sample_t(B, device)
skel = torch.randn(B, 4, 32, 32) * 0.1
img = torch.rand(B, 3, 256, 256) * 2 - 1

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] +
                        [p for p in repa.proj.parameters()], lr=1e-4)
opt.zero_grad()
kw = dict(y_callig=y_callig, y_char=y_char, cond=skel, return_intermediate_layer=8)
ld = diffusion.training_losses(model, x, t, kw)
loss_diff = ld["loss"].mean()
inter = ld.get("intermediate_feats", None)
loss_repa = repa(inter, img) if inter is not None else None
total = loss_diff + 0.1 * loss_repa
print(f"    loss_diff={loss_diff.item():.4f}  loss_repa={loss_repa.item():.4f}  "
      f"inter={None if inter is None else tuple(inter.shape)}")
total.backward()
g = torch.cat([p.grad.flatten() for p in trainable if p.grad is not None])
nnz = (g != 0).sum().item()
print(f"    backward OK, 非零梯度={nnz}/{g.numel()}")
opt.step()
print("[6] SMOKE PASS")