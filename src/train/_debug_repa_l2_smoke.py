# -*- coding: utf-8 -*-
"""CPU 冒烟: REPA-L2 多层(8,11) tuple 路径 + 共享 teacher 验证. 用法: python _debug_repa_l2_smoke.py <base_ckpt>"""
import os, sys
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
model = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=False,
                      **_ctrl_cfg, **_arch).to(device)

_student_dim = main.x_embedder.proj.out_channels
print("[2] REPALoss L2 (共享 teacher) ...")
teacher_obj = None
losses = []
for _i in range(2):
    rl = REPALoss(student_dim=_student_dim, teacher_backbone="dinov2_vits14",
                  teacher_ckpt=None, teacher=teacher_obj).to(device)
    if teacher_obj is None:
        teacher_obj = rl.teacher
    losses.append(rl)
print(f"    teacher 是否共享: {losses[0].teacher is losses[1].teacher}")

diffusion = create_diffusion_or_flow("", diffusion_type="flow", **flow_kwargs_from(args))

print("[3] 前向+反向 (B=2, REPA layers=[8,11]) ...")
B = 2
torch.manual_seed(2)
x = torch.randn(B, 4, 32, 32)
y_callig = torch.randint(0, 1013, (B,))
y_char = torch.randint(0, 35130, (B,))
t = diffusion.sample_t(B, device)
skel = torch.randn(B, 4, 32, 32) * 0.1
img = torch.rand(B, 3, 256, 256) * 2 - 1

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] +
                        [p for pl in losses for p in pl.proj.parameters()], lr=1e-4)
opt.zero_grad()
kw = dict(y_callig=y_callig, y_char=y_char, cond=skel, return_intermediate_layers=[8, 11])
ld = diffusion.training_losses(model, x, t, kw)
loss_diff = ld["loss"].mean()
inter = ld.get("intermediate_feats", None)
print(f"    intermediate 类型: {type(inter).__name__}, 元素数: {len(inter) if isinstance(inter,(list,tuple)) else 'N/A'}")
assert isinstance(inter, tuple), "多层应返回 tuple"
loss_repa = sum(fn(fi, img) for fn, fi in zip(losses, inter)) / len(inter)
total = loss_diff + 0.3 * loss_repa
print(f"    loss_diff={loss_diff.item():.4f}  loss_repa(L2)={loss_repa.item():.4f}")
total.backward()
g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
print(f"    backward OK, 非零梯度={int((g != 0).sum())}/{g.numel()}")
opt.step()
print("[4] SMOKE PASS (REPA-L2 multi-layer + shared teacher OK)")