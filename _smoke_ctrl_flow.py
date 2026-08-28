# -*- coding: utf-8 -*-
"""Smoke test: build s18 flow main model via load_main_model, load 43k ckpt."""
import os, sys, torch
sys.stdout.reconfigure(encoding="utf-8")
_s = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _s)
_c = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "controlnet")
sys.path.insert(0, _c)
from controlnet_dit import load_main_model, ControlNetDiT

ckpt = "5script/results/s18_s_flow_small/20260827-232003-s18-s-flow-small/checkpoints/0043000.pt"
m = load_main_model(
    model_name="DiT-2Cond-S/2", ckpt_path=ckpt, device="cpu",
    num_calligraphers=1011, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128,
    char_embed_dim=384, char_proj_mode="ln_only", freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, use_checkpoint=False)
print("main built:", type(m).__name__, "char_embed_dim=384, proj=", type(m.char_proj).__name__)
ctrl = ControlNetDiT(m, cond_in_channels=1, train_ctrl_only=True)
print("ctrl built; ctrl embed out channels=", ctrl.ctrl_encoder.embed.proj.out_channels)
# sanity forward (flow: velocity)
x = torch.randn(2, 4, 32, 32)
t = torch.full((2,), 500.0)
yc = torch.tensor([0, 1]); yh = torch.tensor([0, 7026])
skel = torch.zeros(2, 1, 256, 256)
with torch.no_grad():
    o = ctrl(x, t, yc, yh, cond=skel)
print("forward out:", tuple(o.shape))
print("OK")
