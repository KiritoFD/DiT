"""单模式训练步基准。用法: python step_bench.py <mode>
mode: none | default | max-autotune | reduce-overhead
每个模式独立进程 (避免 Dynamo 状态串扰), 共享 inductor 磁盘缓存。
"""
import sys, os, time, json
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/root/.cache/torch/inductor")
import torch
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import DiT_2Cond_models
from torch.utils.flop_counter import FlopCounterMode

mode = sys.argv[1] if len(sys.argv) > 1 else "default"
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
PEAK, B = 172e12, 240

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

m = DiT_2Cond_models["DiT-2Cond-S/2"](**KW).to(dev)
with torch.no_grad():
    with FlopCounterMode(display=False) as fc:
        m(x, t, yc, ych, g=g)
TOT = fc.get_total_flops() * 3.0

t0 = time.perf_counter()
mc = m if mode == "none" else torch.compile(m, mode=mode)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.02)


def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mc(x, t, yc, ych, g=g)
    torch.nn.functional.mse_loss(out.float(), tgt).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()


for _ in range(6):
    step()
torch.cuda.synchronize()
t_first = time.perf_counter() - t0          # 含首次编译
N = 30
t0 = time.perf_counter()
for _ in range(N):
    step()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N
r = dict(mode=mode, ms_per_step=round(dt * 1e3, 1),
         tflops=round(TOT / dt / 1e12, 1),
         pct_of_peak=round(TOT / dt / PEAK * 100, 1),
         startup_compile_s=round(t_first, 1))
print("RESULT " + json.dumps(r), flush=True)
