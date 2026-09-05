# -*- coding: utf-8 -*-
"""hw_gemm.py — GEMM 线程扩展性扫描 (单形状单线程数, 由驱动循环调用).

用法: taskset -c <T个核> python hw_gemm.py <threads>
形状: square(4096^3) 峰值参照 / qkv(8192,384)@(384,1152) / mlp_up(8192,384)@(384,1536)
      / mlp_down(8192,1536)@(1536,384) —— 与 DiT-S/2 实际 GEMM 同形
"""
import os, sys, time
sys.path.insert(0, "/root/Workspace/xy/DiT")
NT = int(sys.argv[1])
import torch
torch.set_num_threads(NT)


def bench(f, n):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n


shapes = [
    ("square4096", torch.randn(4096, 4096), torch.randn(4096, 4096), 6),
    ("qkv_K384", torch.randn(8192, 384), torch.randn(384, 1152), 40),
    ("mlpup_K384", torch.randn(8192, 384), torch.randn(384, 1536), 40),
    ("mlpdown_K1536", torch.randn(8192, 1536), torch.randn(1536, 384), 25),
]
out = [f"GEMM threads={NT}"]
for name, A, B, reps in shapes:
    flops = 2 * A.shape[0] * A.shape[1] * B.shape[1]
    dt = bench(lambda: A @ B, reps)
    out.append(f"  {name:16s} {dt*1000:8.2f} ms  {flops/dt/1e12:6.3f} TFLOPS")
    del A, B
print("\n".join(out), flush=True)
