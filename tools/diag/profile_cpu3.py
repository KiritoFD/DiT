# -*- coding: utf-8 -*-
"""profile_cpu3.py — 三个决定性小实验: 分配churn / attention路径 / out=复用.

T1: glibc malloc 阈值 (mmap churn) 对 forward 的影响 —— 需在环境变量下启动本进程
T2: attn_impl sdpa vs eager 的 forward 时间
T3: 同形状 mm: fresh-alloc vs out= 预分配缓冲
"""
import os, sys, time
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
import torch
torch.set_num_threads(NT)
thp = open("/sys/kernel/mm/transparent_hugepage/enabled").read().strip()
print(f"threads={NT} THP={thp} MMAP_TH={os.environ.get('MALLOC_MMAP_THRESHOLD_','unset')}", flush=True)


def bench(f, n=5):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n


print("\n== T3 mm fresh vs out= 复用 ==", flush=True)
A = torch.randn(8192, 384)
W = torch.randn(384, 1152)
OUT = torch.empty(8192, 1152)
dt_fresh = bench(lambda: A @ W, 30)
dt_out = bench(lambda: torch.mm(A, W, out=OUT), 30)
print(f"  fresh-alloc mm: {dt_fresh*1000:.1f} ms | out= 复用: {dt_out*1000:.1f} ms "
      f"(分配开销占比 {(dt_fresh-dt_out)/dt_fresh*100:.0f}%)", flush=True)

print("\n== T2 attn_impl sdpa vs eager ==", flush=True)
from src.model.controlnet import load_main_model, ControlNetDiT
ARCH = dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
            rope_theta=100.0, attn_impl="sdpa")
COMMON = dict(device=torch.device("cpu"), num_calligraphers=1013,
              num_characters=35130, condition_fusion="factorized_add",
              callig_embed_dim=128, char_embed_dim=384, char_proj_mode="mlp",
              freeze_char_table=True, learn_sigma=False, **ARCH)
x = torch.randn(32, 4, 32, 32)
t = torch.full((32,), 500.0)
yc = torch.randint(0, 1000, (16,)).repeat(2)
yh = torch.randint(0, 30000, (16,)).repeat(2)
sk = torch.randn(16, 4, 32, 32).repeat(2, 1, 1, 1)
for impl in ("sdpa", "eager"):
    main = load_main_model(ckpt_path="5script/results/v8_3stage/A_main_final.pt",
                           attn_impl=impl, **{k: v for k, v in COMMON.items() if k != "attn_impl"})
    main.eval()
    c = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True,
                      injection="modulate", null_cond="gaussian", attn_impl=impl)
    ckd = torch.load("5script/results/v8_3stage/v8b/20260902-234912-v8b-s31-ctrl/checkpoints/0035000.pt",
                     map_location="cpu", weights_only=False)
    sd = {k: v for k, v in (ckd.get("ema") or ckd.get("ctrl")).items()
          if not k.startswith("main.") and not k.startswith("_orig_mod.main.")}
    c.load_state_dict(sd, strict=False)
    c.eval()
    with torch.no_grad():
        dt = bench(lambda: c.forward_with_cfg(x, t, yc, yh, cfg_scale=0.7, cond=sk), 5)
    print(f"  attn_impl={impl:5s}: forward {dt*1000:.0f} ms", flush=True)
    del main, c
print("PROFILE3_DONE", flush=True)
