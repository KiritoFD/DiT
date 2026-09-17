"""infra 基准 (GPU 独占): 定出真实可达上限, 并拆解一个训练步的时间去向。"""
import sys, time, torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models
from torch.utils.flop_counter import FlopCounterMode

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,power.limit,temperature.gpu",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())


def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


print("\n" + "=" * 80)
print("A. 峰值: 大 GEMM (burst vs sustained)")
for n, it in [(4096, 30), (8192, 20)]:
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    dt = bench(lambda: a @ b, it)
    print(f"   {n}^3  {2*n**3/dt/1e12:7.1f} TFLOPs/s   ({dt*1e3:.2f} ms)")
# sustained: 连续算 8 秒
a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize(); t0 = time.perf_counter(); cnt = 0
while time.perf_counter() - t0 < 8.0:
    a @ a; cnt += 1
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / cnt
print(f"   8192^3 持续 8s: {2*8192**3/dt/1e12:7.1f} TFLOPs/s  <- 持续上限(含降频)")
print("   " + subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())

print("\n" + "=" * 80)
print("B. 我们模型的 GEMM 形状 (B=240)")
B, T, H = 240, 256, 384
M = B * T
for nm, m_, n_, k_ in [("qkv    h->3h", M, 3 * H, H), ("proj   h->h", M, H, H),
                       ("mlp fc1 h->8h/3", M, 1024, H), ("mlp fc2 8h/3->h", M, H, 1024)]:
    aa = torch.randn(m_, k_, device=dev, dtype=torch.bfloat16)
    bb = torch.randn(k_, n_, device=dev, dtype=torch.bfloat16)
    dt = bench(lambda: aa @ bb, 200)
    print(f"   {nm:18s} ({m_}x{k_})@({k_}x{n_})  {2*m_*n_*k_/dt/1e12:7.1f} TFLOPs/s")

print("\n" + "=" * 80)
print("C. 完整训练步 (fwd+bwd+AdamW), 与真实日志对照")
kw = dict(input_size=32, in_channels=4, num_calligraphers=52, callig_embed_dim=128,
          use_char_cond=False, use_glyph_cond=True, glyph_scale_init=0.6,
          glyph_embedder_depth=2, glyph_embedder_sep=False, glyph_inject_layers=4,
          glyph_inject_mode="adaln", condition_fusion="factorized_cat",
          glyph_vec_cond=True, norm_type="rms", mlp_type="swiglu", qk_norm=True,
          rope=True, attn_impl="sdpa", image_channels=4, learn_sigma=False)
m = DiT_2Cond_models["DiT-2Cond-S/2"](**kw).to(dev)
x = torch.randn(B, 4, 32, 32, device=dev); t = torch.rand(B, device=dev)
yc = torch.randint(0, 52, (B,), device=dev); ych = torch.zeros(B, dtype=torch.long, device=dev)
g = torch.randn(B, 4, 32, 32, device=dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.02)


def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x, t, yc, ych, g=g)
    loss = torch.nn.functional.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()


dt = bench(step, 30, warmup=5)
with torch.no_grad():
    with FlopCounterMode(display=False) as fc:
        m(x, t, yc, ych, g=g)
fwd = fc.get_total_flops()          # 已经是整个 batch 的前向
tot = fwd * 3.0                     # fwd+bwd ~ 3x
PEAK = 172e12
print(f"   step FLOPs = {tot/1e12:.3f} TFLOPs (fwd {fwd/1e12:.3f} x3)")
print(f"   未 compile: {dt*1e3:6.1f} ms/step -> {tot/dt/1e12:6.1f} TFLOPs/s "
      f"({tot/dt/PEAK*100:.1f}% of {PEAK/1e12:.0f}T 峰值)")
print(f"   真实训练   : 178.2 ms/step (5.61 step/s) -> {tot/0.1782/1e12:6.1f} TFLOPs/s "
      f"({tot/0.1782/PEAK*100:.1f}%)")
print(f"   -> torch.compile 已带来 {dt*1e3/178.2:.2f}x 提速")
print(f"   -> 与 GEMM 能力(150T)相比, compile 后仍有 {150/(tot/0.1782/1e12):.2f}x 空间")

print("\n" + "=" * 80)
print("C2. 逐 kernel 时间拆解 (compile 后的真实训练步)")
from torch.profiler import profile, ProfilerActivity


def get_dev_us(e):
    for attr in ("device_time_total", "self_device_time_total",
                 "cuda_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


mc = torch.compile(m, mode="default")
for _ in range(8):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()
agg = {}
for e in prof.key_averages():
    t = get_dev_us(e)
    if t <= 0:
        continue
    nm = e.key.split("(")[0]
    for pfx in ["aten::", "void ", "nvjet_", "cutlass::", "sm90_xmma_", "sm80_xmma_"]:
        nm = nm.replace(pfx, "")
    nm = nm[:44]
    agg[nm] = agg.get(nm, 0.0) + t
total_us = sum(agg.values())
print(f"   kernel 合计 {total_us/1e3/10:.1f} ms/step  (占真实 178ms 的 "
      f"{total_us/1e3/10/178.2*100:.0f}%)")
for nm, us in sorted(agg.items(), key=lambda kv: -kv[1])[:16]:
    print(f"     {nm:46s} {us/1e3/10:7.2f} ms  {100*us/total_us:5.1f}%")



print("\n" + "=" * 80)
print("D. torch.compile 模式对比")
for mode in ["default", "max-autotune", "reduce-overhead"]:
    try:
        m2 = DiT_2Cond_models["DiT-2Cond-S/2"](**kw).to(dev)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.02)
        m2c = torch.compile(m2, mode=mode)

        def step2():
            opt2.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m2c(x, t, yc, ych, g=g)
            torch.nn.functional.mse_loss(out.float(), torch.zeros_like(out)).backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt2.step()

        dt2 = bench(step2, 30, warmup=8)
        print(f"   compile({mode:15s}) = {dt2*1e3:6.1f} ms/step -> {tot/dt2/1e12:5.1f} TFLOPs/s "
              f"(相对 default {dt2/dt:.3f}x)")
        del m2, m2c, opt2
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   compile({mode}) FAILED: {type(e).__name__}: {str(e)[:120]}")
