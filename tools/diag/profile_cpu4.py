# -*- coding: utf-8 -*-
"""profile_cpu4.py — 收尾实验: attention 路径 / in-module Linear / 非连续 / 分配churn量化.

A attention 微基准 (32,6,256,64): sdpa-flash vs eager bmm+softmax vs 合头 mm 重构
B 12块 nn.Linear 堆栈 (实际 GEMM 形状, 连续输入): in-module mm 真实速率
C 非连续输入 mm 对照
D 每次前向的分配总量推算 (逐 op 输出字节)
"""
import os, sys, time
sys.path.insert(0, "/root/Workspace/xy/DiT")
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
import torch
import torch.nn as nn
torch.set_num_threads(NT)
print(f"threads={NT}", flush=True)


def bench(f, n=10):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n


print("\n== A attention 路径 (B=32, H=6, T=256, Dh=64) ==", flush=True)
q = torch.randn(32, 6, 256, 64)
k = torch.randn(32, 6, 256, 64)
v = torch.randn(32, 6, 256, 64)
fl = 2 * 2 * 32 * 6 * 256 * 256 * 64  # QK + AV

with torch.no_grad():
    dt = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), 10)
    print(f"  sdpa (flash CPU):   {dt*1000:7.1f} ms  {fl/dt/1e12:.3f} TFLOPS", flush=True)

    def eager():
        s = (q @ k.transpose(-1, -2)) / 8.0
        s = torch.softmax(s, dim=-1)
        return s @ v
    dt = bench(eager, 10)
    print(f"  eager bmm+softmax:  {dt*1000:7.1f} ms  {fl/dt/1e12:.3f} TFLOPS", flush=True)

    # 合头重构: (32*256, 384) @ (384, 256) -> softmax -> @ (256,384): 大 GEMM 形状
    qm = q.permute(0, 2, 1, 3).reshape(32 * 256, 384)
    km = k.permute(0, 2, 1, 3).reshape(32 * 256, 384)
    vm = v.permute(0, 2, 1, 3).reshape(32 * 256, 384)

    def reform():
        s = (qm @ km.transpose(0, 1)).view(32, 256, 256) / 8.0
        s = torch.softmax(s, dim=-1)
        s = s.reshape(32 * 256, 256)
        return s @ vm
    dt = bench(reform, 10)
    print(f"  合头 mm 重构:       {dt*1000:7.1f} ms  {fl/dt/1e12:.3f} TFLOPS  (数值=逐头分组, 需分块掩码时不同)", flush=True)

print("\n== B 12块 nn.Linear 堆栈 (实际形状) ==", flush=True)
lins = nn.ModuleList([
    nn.Linear(384, 1152), nn.Linear(1152, 384),
    nn.Linear(384, 1536), nn.Linear(1536, 384)] * 3)
xs = torch.randn(8192, 384)
gflops = 2 * 8192 * (384 * 1152 + 1152 * 384 + 384 * 1536 + 1536 * 384) * 3 / 1e9


def stack():
    h = xs
    with torch.no_grad():
        for l in lins:
            h = torch.nn.functional.silu(l(h))
    return h
with torch.no_grad():
    dt = bench(stack, 10)
print(f"  12块堆栈: {dt*1000:.0f} ms -> {gflops/dt/1e3:.2f} TFLOPS (raw A@W 同形状 32t: 0.55-0.86)", flush=True)

print("\n== C 非连续输入 mm ==", flush=True)
big = torch.randn(384, 8192)
A_nc = big.t()          # (8192,384) 非连续
A_c = big.t().contiguous()
W = torch.randn(384, 1152)
dt_nc = bench(lambda: A_nc @ W, 20)
dt_c = bench(lambda: A_c @ W, 20)
print(f"  非连续 A: {dt_nc*1000:.1f} ms | 连续 A: {dt_c*1000:.1f} ms | 比值 {dt_nc/dt_c:.2f}x", flush=True)

print("\n== D 前向分配量推算 ==", flush=True)
rows = 8192
per_block = {
    "x(ln后)": rows * 384 * 4, "qkv": rows * 1152 * 4, "attn_score": 32 * 6 * 256 * 256 * 4,
    "attn_out": rows * 384 * 4, "proj": rows * 384 * 4,
    "mod_s+t": 2 * rows * 384 * 4, "gate": rows * 1536 * 4, "up": rows * 1536 * 4,
    "silu": rows * 1536 * 4, "down": rows * 384 * 4}
tot = sum(per_block.values()) * 12 * 2  # main + ctrl encoder
print(f"  每 forward 新分配 ≈ {tot/1e9:.1f} GB (24 个块 × 中间张量)", flush=True)
print(f"  按 alloc+touch 实测 0.195 ms/MB (64MB=12.5ms) -> 纯分配/页错误 ≈ {tot/1e6*0.195/1e3:.2f} s/forward", flush=True)
print("PROFILE4_DONE", flush=True)
