# -*- coding: utf-8 -*-
"""profile_cpu7.py — 按 (op, 输入形状) 聚合的前向热点 + 线程数对照."""
import os, sys, time
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
import torch
torch.set_num_threads(NT)

from src.model.controlnet import load_main_model, ControlNetDiT
ARCH = dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
            rope_theta=100.0, attn_impl="sdpa")
COMMON = dict(device=torch.device("cpu"), num_calligraphers=1013,
              num_characters=35130, condition_fusion="factorized_add",
              callig_embed_dim=128, char_embed_dim=384, char_proj_mode="mlp",
              freeze_char_table=True, learn_sigma=False, **ARCH)
main = load_main_model(ckpt_path="5script/results/v8_3stage/A_main_final.pt", **COMMON)
main.eval()
c = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True,
                  injection="modulate", null_cond="gaussian")
ckd = torch.load("5script/results/v8_3stage/v8b/20260902-234912-v8b-s31-ctrl/checkpoints/0035000.pt",
                 map_location="cpu", weights_only=False)
sd = {k: v for k, v in (ckd.get("ema") or ckd.get("ctrl")).items()
      if not k.startswith("main.") and not k.startswith("_orig_mod.main.")}
c.load_state_dict(sd, strict=False)
c.eval()

x = torch.randn(32, 4, 32, 32)
t = torch.full((32,), 500.0)
yc = torch.randint(0, 1000, (16,)).repeat(2)
yh = torch.randint(0, 30000, (16,)).repeat(2)
sk = torch.randn(16, 4, 32, 32).repeat(2, 1, 1, 1)

from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    for _ in range(3):
        c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(2):
            c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)

ka = prof.key_averages(group_by_input_shape=True)
rows = sorted(ka, key=lambda k: -k.self_cpu_time_total)[:18]
tot = sum(k.self_cpu_time_total for k in ka)
print(f"threads={NT} 2次forward 总self时间 {tot/1e3:.0f} ms -> {tot/2e3:.0f} ms/fwd", flush=True)
for k in rows:
    print(f"  {k.key[:42]:42s} {str(k.input_shapes)[:44]:44s} "
          f"n={k.count:>3} {k.self_cpu_time_total/2e3:7.1f} ms/fwd", flush=True)
print("P7_DONE", flush=True)
