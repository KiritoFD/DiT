"""聚焦: profile compile 后的训练步, 并对比 compile 模式。"""
import sys, time, torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models
from torch.utils.flop_counter import FlopCounterMode

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
PEAK = 172e12
B = 240

KW = dict(input_size=32, in_channels=4, num_calligraphers=52, callig_embed_dim=128,
          use_char_cond=False, use_glyph_cond=True, glyph_scale_init=0.6,
          glyph_embedder_depth=2, glyph_embedder_sep=False, glyph_inject_layers=4,
          glyph_inject_mode="adaln", condition_fusion="factorized_cat",
          glyph_vec_cond=True, norm_type="rms", mlp_type="swiglu", qk_norm=True,
          rope=True, attn_impl="sdpa", image_channels=4, learn_sigma=False)

x = torch.randn(B, 4, 32, 32, device=dev)
t = torch.rand(B, device=dev)
yc = torch.randint(0, 52, (B,), device=dev)
ych = torch.zeros(B, dtype=torch.long, device=dev)
g = torch.randn(B, 4, 32, 32, device=dev)
tgt = torch.zeros(B, 4, 32, 32, device=dev)

m0 = DiT_2Cond_models["DiT-2Cond-S/2"](**KW).to(dev)
with torch.no_grad():
    with FlopCounterMode(display=False) as fc:
        m0(x, t, yc, ych, g=g)
TOT = fc.get_total_flops() * 3.0
print(f"step FLOPs (fwd+bwd) = {TOT/1e12:.3f} TFLOPs | 峰值 {PEAK/1e12:.0f}T")
del m0
torch.cuda.empty_cache()


def make(mode=None):
    torch._dynamo.reset()
    m = DiT_2Cond_models["DiT-2Cond-S/2"](**KW).to(dev)
    mc = torch.compile(m, mode=mode) if mode else m
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.02)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mc(x, t, yc, ych, g=g)
        torch.nn.functional.mse_loss(out.float(), tgt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m, step


def bench(step, iters=30, warmup=6):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


print("\n" + "=" * 78)
print("compile 模式对比")
res = {}
for mode in [None, "default", "max-autotune", "reduce-overhead"]:
    tag = mode or "none"
    try:
        m, step = make(mode)
        dt = bench(step)
        res[tag] = dt
        print(f"   {tag:16s} {dt*1e3:6.1f} ms/step  {TOT/dt/1e12:6.1f} TFLOPs/s  "
              f"({TOT/dt/PEAK*100:4.1f}% of peak)")
        del m, step
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   {tag:16s} FAILED: {type(e).__name__}: {str(e)[:100]}")

best = min(res, key=res.get) if res else None
if best:
    print(f"\n   最优 = {best} ({res[best]*1e3:.1f} ms) -> 相对 default "
          f"{res.get('default', res[best])/res[best]:.3f}x")
    print(f"   真实训练日志 = 178.2 ms/step -> 与最优差 "
          f"{178.2/res[best]*1e3 - 1000:.0f} ms 的非模型开销")

print("\n" + "=" * 78)
print("逐 kernel 拆解 (compile default, 只统计 GPU kernel 时间)")
from torch.profiler import profile, ProfilerActivity

m, step = make("default")
for _ in range(6):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()


def gus(e):
    for a in ("device_time_total", "self_device_time_total",
              "cuda_time_total", "self_cuda_time_total"):
        v = getattr(e, a, None)
        if v is not None:
            return float(v)
    return 0.0


agg = {}
for e in prof.key_averages():
    tt = gus(e)
    if tt <= 0:
        continue
    nm = e.key.split("(")[0]
    for p in ["aten::", "void ", "nvjet_", "cutlass::", "sm90_xmma_", "sm80_xmma_",
              "ampere_", "triton_"]:
        nm = nm.replace(p, "")
    agg[nm[:44]] = agg.get(nm[:44], 0.0) + tt
tot_us = sum(agg.values())
print(f"   kernel 合计 {tot_us/1e3/10:.1f} ms/step")
for nm, us in sorted(agg.items(), key=lambda kv: -kv[1])[:14]:
    print(f"     {nm:46s} {us/1e3/10:7.2f} ms  {100*us/tot_us:5.1f}%")
