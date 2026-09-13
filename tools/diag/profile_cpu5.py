# -*- coding: utf-8 -*-
"""profile_cpu5.py — THP 后的干净对照: forward 基线 / torch.compile / Linear 堆栈.

用法: taskset -c 32-63 python profile_cpu5.py 32
"""
import os, sys, time
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
import torch
torch.set_num_threads(NT)


def bench(f, n=8):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n


from src.model.controlnet import load_main_model, ControlNetDiT
ARCH = dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
            rope_theta=100.0, attn_impl="sdpa")
COMMON = dict(device=torch.device("cpu"), num_calligraphers=1013,
              num_characters=35130, condition_fusion="factorized_add",
              callig_embed_dim=128, char_embed_dim=384, char_proj_mode="mlp",
              freeze_char_table=True, learn_sigma=False, **ARCH)
main = load_main_model(ckpt_path="assets/results/v8_3stage/A_main_final.pt", **COMMON)
main.eval()
c = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True,
                  injection="modulate", null_cond="gaussian")
ckd = torch.load("assets/results/v8_3stage/v8b/20260902-234912-v8b-s31-ctrl/checkpoints/0035000.pt",
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

with torch.no_grad():
    dt = bench(lambda: c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk), 10)
print(f"THP 后 eager forward (32行): {dt*1000:.0f} ms", flush=True)

# Linear 堆栈: in-module mm 速率 vs raw A@W (定位剩余 gap)
import torch.nn as nn
lins = [nn.Linear(384, 1152), nn.Linear(1152, 384),
        nn.Linear(384, 1536), nn.Linear(1536, 384)] * 3
xs = torch.randn(8192, 384)


def stack():
    h = xs
    with torch.no_grad():
        for l in lins:
            h = torch.nn.functional.silu(l(h))
    return h
with torch.no_grad():
    dts = bench(stack, 10)
gflops = 2 * 8192 * (384*1152 + 1152*384 + 384*1536 + 1536*384) * 3 / 1e9
print(f"12块 Linear 堆栈 (含 silu): {dts*1000:.0f} ms -> {gflops/dts/1e3:.2f} TFLOPS "
      f"(raw A@W 32t: 0.55-0.86)", flush=True)

# torch.compile 对照
try:
    cc = torch.compile(c, dynamic=False)
    with torch.no_grad():
        t0 = time.time()
        _ = cc.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)
        print(f"compile 首推理: {time.time()-t0:.0f} s (含编译)", flush=True)
        dtc = bench(lambda: cc.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk), 10)
    print(f"compiled forward: {dtc*1000:.0f} ms (加速 {dt/dtc:.2f}x)", flush=True)
except Exception as e:
    print(f"compile 失败: {type(e).__name__}: {str(e)[:150]}", flush=True)
print("P5_DONE", flush=True)
