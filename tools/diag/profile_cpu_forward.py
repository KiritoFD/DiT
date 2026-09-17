# -*- coding: utf-8 -*-
"""profile_cpu_forward.py — 定位 CPU eval 瓶颈: BLAS 健康 vs eager elementwise 开销.

输出三段:
  S1 BLAS 基准: 大方阵 GEMM (峰值参照) + 实际形状 GEMM (M=8192,K=384,N∈{1152,1536})
  S2 单次 forward 分解: torch.profiler 按 op 类聚合 (mm/addmm vs norm/elementwise/softmax)
  S3 compile 对照: torch.compile 前后 per-forward 时间 (若可用)
"""
import os, sys, time
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
import torch
torch.set_num_threads(NT)
print(f"threads={NT} | blas={torch.backends.cpu.get_cpu_capability() if hasattr(torch.backends.cpu,'get_cpu_capability') else '?'}", flush=True)


def tflops(f, flops, n=1):
    t0 = time.time()
    for _ in range(n):
        f()
    dt = (time.time() - t0) / n
    return flops / dt / 1e12, dt


print("\n== S1 BLAS 健康 ==", flush=True)
A = torch.randn(4096, 4096)
Bm = torch.randn(4096, 4096)
_ = A @ Bm  # warm
tf, dt = tflops(lambda: A @ Bm, 2 * 4096**3, 10)
print(f"  square 4096^3: {tf:.2f} TFLOPS ({dt*1000:.0f} ms)  <- 峰值参照 (单socket峰~2.9)", flush=True)
del A, Bm

X = torch.randn(8192, 384)
Wq = torch.randn(384, 1152)
Wm = torch.randn(384, 1536)
_ = X @ Wq
tf1, _ = tflops(lambda: X @ Wq, 2 * 8192 * 384 * 1152, 50)
tf2, _ = tflops(lambda: X @ Wm, 2 * 8192 * 384 * 1536, 50)
print(f"  qkv 形状 (8192,384)@(384,1152): {tf1:.2f} TFLOPS", flush=True)
print(f"  mlp 形状 (8192,384)@(384,1536): {tf2:.2f} TFLOPS", flush=True)
G = X @ Wm
tf3, _ = tflops(lambda: G @ Wm.t(), 2 * 8192 * 1536 * 384, 50)
print(f"  mlp down (8192,1536)@(1536,384): {tf3:.2f} TFLOPS", flush=True)
del X, Wq, Wm, G

print("\n== S2 单次 forward 分解 (ctrl wrapper, 32 行 = 16样本 CFG 拼批) ==", flush=True)
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
    _ = c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)  # warm
    tf, dt = tflops(lambda: c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk),
                    77e9, 10)  # 77 GFLOPs 理论 (32行×256tok, main+ctrl)
print(f"  eager forward: {dt*1000:.0f} ms -> {tf:.2f} TFLOPS", flush=True)

from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    for _ in range(3):
        _ = c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            _ = c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)
ka = prof.key_averages()
gemm, elem, norm, soft, other = 0.0, 0.0, 0.0, 0.0, 0.0
for k in ka:
    dt_op = getattr(k, "self_device_time_total", 0) or 0
    dt_op = (k.self_cpu_time_total or 0) / 1000.0 / 3  # 秒, 除次数
    name = k.key
    if any(s in name for s in ("mm", "addmm", "bmm", "linear", "conv")):
        gemm += dt_op
    elif any(s in name for s in ("layer_norm", "rms", "group_norm")):
        norm += dt_op
    elif "softmax" in name:
        soft += dt_op
    elif any(s in name for s in ("mul", "add", "silu", "exp", "div", "sub", "cat", "copy", "view", "reshape", "neg", "pow", "mean", "sum")):
        elem += dt_op
    else:
        other += dt_op
tot = gemm + elem + norm + soft + other
print(f"  3次 forward op 时间分布: GEMM {gemm:.3f}s | elementwise {elem:.3f}s | "
      f"norm {norm:.3f}s | softmax {soft:.3f}s | other {other:.3f}s (合计 {tot:.3f}s)", flush=True)
print("  top10:", flush=True)
for k in sorted(ka, key=lambda k: -k.self_cpu_time_total)[:10]:
    print(f"    {k.key[:60]:60s} {k.self_cpu_time_total/3000:.1f} ms/fwd", flush=True)

print("\n== S3 torch.compile 对照 ==", flush=True)
try:
    cc = torch.compile(c, dynamic=False)
    with torch.no_grad():
        _ = cc.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)  # compile
    t0 = time.time()
    _ = cc.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk)
    print(f"  compile 首推理后单次: {(time.time()-t0)*1000:.0f} ms", flush=True)
    tfc, dtc = tflops(lambda: cc.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk),
                      77e9, 10)
    print(f"  compiled forward: {dtc*1000:.0f} ms -> {tfc:.2f} TFLOPS "
          f"(加速 {dt/dtc:.2f}x)", flush=True)
except Exception as e:
    print(f"  compile 失败: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("PROFILE_DONE", flush=True)
