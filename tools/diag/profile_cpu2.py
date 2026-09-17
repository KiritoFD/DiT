# -*- coding: utf-8 -*-
"""profile_cpu2.py — 清场后的瓶颈定位 v2.

前置: 杀掉其它 bench/python 重负载, 只留训练。
  S1' GEMM 线程扩展性: 8/16/32 线程 (OMP_PROC_BIND=close, 检查 CCX 拓扑墙)
  S2' 干净单 forward op 分布 (无干扰)
  S3' heun_batch 对比 (每步 6B vs 4B 行, 在 CPU 上到底差多少)
"""
import os, sys, time
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
import torch
torch.set_num_threads(NT)
print(f"threads={NT} blas_cap={torch.backends.cpu.get_cpu_capability()}", flush=True)


def bench(f, n):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n


print("\n== S1' GEMM 线程扩展性 ==", flush=True)
A = torch.randn(8192, 384)
W = torch.randn(384, 1152)
dt = bench(lambda: A @ W, 30)
print(f"  {NT}t qkv(8192,384)@(384,1152): {dt*1000:.0f} ms = {2*8192*384*1152/dt/1e12:.2f} TFLOPS", flush=True)
B4 = torch.randn(4096, 4096)
dt = bench(lambda: B4 @ B4, 8)
print(f"  {NT}t square(4096^3):              {dt*1000:.0f} ms = {2*4096**3/dt/1e12:.2f} TFLOPS", flush=True)

print("\n== S2' 干净单 forward 分布 ==", flush=True)
from src.model.legacy.controlnet import load_main_model, ControlNetDiT
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
    dt = bench(lambda: c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk), 5)
    print(f"  eager CFG forward (32行): {dt*1000:.0f} ms -> {77e9/dt/1e12:.2f} TFLOPS", flush=True)
    # 纯 main (无 ctrl encoder, 无 CFG) 的对照
    dt2 = bench(lambda: main(x[:16], t[:16] * 1000, yc[:16], yh[:16]), 5)
    print(f"  eager main-only forward (16行): {dt2*1000:.0f} ms -> {38.5e9/dt2/1e12:.2f} TFLOPS", flush=True)

from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    for _ in range(3):
        _ = c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _ = c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)
ka = prof.key_averages()
rows = sorted(ka, key=lambda k: -k.self_cpu_time_total)[:14]
gemm = sum(k.self_cpu_time_total for k in ka if any(s in k.key for s in ("mm", "addmm", "bmm", "linear")))
elem = sum(k.self_cpu_time_total for k in ka if any(s in k.key for s in ("mul", "add", "silu", "exp", "div", "sub", "cat", "copy", "pow", "sum", "mean", "neg", "clamp")))
print(f"  GEMM 合计 {gemm/1e3:.0f} ms | elementwise 合计 {elem/1e3:.0f} ms (单 forward)", flush=True)
for k in rows:
    print(f"    {k.key[:56]:56s} {k.self_cpu_time_total/1e3:7.1f} ms", flush=True)

print("\n== S3' 逐 stage Heun vs heun_batch (16样本臂, 3 NFE) ==", flush=True)
from src.utils.cpu_sampler import heun_sample_cpu
from src.loss import create_diffusion_or_flow
noise = torch.randn(16, 4, 32, 32)
conds = [(1, 100)] * 16
skel = torch.randn(16, 4, 32, 32)
for hb in (False, True):
    fl = create_diffusion_or_flow("2", diffusion_type="flow", sampler="heun", heun_batch=hb,
                                  t_sampler="logit_normal", t_mean=0.0, t_std=1.0, shift=1.0)
    t0 = time.time()
    _ = fl.ddim_sample_loop(c, (16, 4, 32, 32), noise.clone(),
                            model_kwargs=dict(y_callig=torch.randint(0, 1000, (16,)),
                                              y_char=torch.randint(0, 30000, (16,)),
                                              cond=skel), device=torch.device("cpu"))
    print(f"  heun_batch={hb}: 2步×16样本 = {time.time()-t0:.2f}s (每 NFE {(time.time()-t0)/4:.3f}s)", flush=True)
print("PROFILE2_DONE", flush=True)
