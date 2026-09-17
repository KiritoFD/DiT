"""测量 4090 上"实际可达"的 bf16 吞吐，用来定 MFU 的真实上限。

分三层:
  1) 大方阵 GEMM (8192^3)      —— 该卡的实践峰值 (cuBLAS 最优形状)
  2) 我们模型里的真实 GEMM 形状 —— 看形状是否吃亏
  3) 完整模型的 fwd+bwd        —— 看端到端达成率
"""
import sys, time, torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
print("matmul precision:", torch.get_float32_matmul_precision())


def bench_gemm(M, N, K, dtype=torch.bfloat16, iters=50, warmup=10):
    a = torch.randn(M, K, device=dev, dtype=dtype)
    b = torch.randn(K, N, device=dev, dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2.0 * M * N * K
    return flops / dt / 1e12, dt * 1e3


print("\n" + "=" * 78)
print("1) 大方阵 GEMM (cuBLAS 最优形状) —— 这是该卡的实践上限")
for n in [4096, 8192, 16384]:
    tf, ms = bench_gemm(n, n, n, iters=20)
    print(f"   {n:>6}^3   {tf:7.1f} TFLOPs/s   ({ms:.2f} ms/次)")

print("\n" + "=" * 78)
print("2) 我们模型的真实 GEMM 形状 (B=240, N=256 tokens 已展平进 M)")
B, T, H = 240, 256, 384
M = B * T
shapes = [
    ("qkv      h->3h ", M, 3 * H, H),
    ("attn.proj h->h  ", M, H, H),
    ("mlp fc1  h->8h/3", M, 1024, H),
    ("mlp fc2  8h/3->h", M, H, 1024),
    ("final   h->16   ", M, 16, H),
]
for nm, m_, n_, k_ in shapes:
    tf, ms = bench_gemm(m_, n_, k_, iters=200)
    print(f"   {nm}  ({m_}x{k_})@({k_}x{n_})  {tf:7.1f} TFLOPs/s")

print("\n" + "=" * 78)
print("3) 完整模型 fwd+bwd 端到端")
from torch.utils.flop_counter import FlopCounterMode
kw = dict(input_size=32, in_channels=4, num_calligraphers=52, callig_embed_dim=128,
          use_char_cond=False, use_glyph_cond=True, glyph_scale_init=0.6,
          glyph_embedder_depth=2, glyph_embedder_sep=False, glyph_inject_layers=4,
          glyph_inject_mode="adaln", condition_fusion="factorized_cat",
          glyph_vec_cond=True, norm_type="rms", mlp_type="swiglu", qk_norm=True,
          rope=True, attn_impl="sdpa", image_channels=4, learn_sigma=False)
m = DiT_2Cond_models["DiT-2Cond-S/2"](**kw).to(dev)
x = torch.randn(B, 4, 32, 32, device=dev)
t = torch.rand(B, device=dev)
yc = torch.randint(0, 52, (B,), device=dev)
g = torch.randn(B, 4, 32, 32, device=dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, t, yc, torch.zeros_like(yc), g=g).float().mean()
    loss.backward(); opt.step()
torch.cuda.synchronize()
N_IT = 20
t0 = time.perf_counter()
for _ in range(N_IT):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, t, yc, torch.zeros_like(yc), g=g).float().mean()
    loss.backward(); opt.step()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N_IT
with torch.no_grad():
    with FlopCounterMode(display=False) as fc:
        m(x, t, yc, torch.zeros_like(yc), g=g)
fwd = fc.get_total_flops()
tot = fwd * 3.0
print(f"   前向 FLOPs/sample = {fwd/1e9:.3f} G  (x{B} = {fwd*B/1e12:.3f} TFLOPs/step)")
print(f"   fwd+bwd ~ 3x     = {tot*B/1e12:.3f} TFLOPs/step")
print(f"   实测 {dt*1e3:.1f} ms/step -> {tot*B/dt/1e12:.1f} TFLOPs/s")
