# -*- coding: utf-8 -*-
"""hw_mem.py — 内存带宽 (stream-like triad) + 分配/页错误 microbench.

用法: taskset -c <cores> python hw_mem.py <threads>
"""
import os, sys, time
sys.path.insert(0, "/root/Workspace/xy/DiT")
NT = int(sys.argv[1])
import torch
torch.set_num_threads(NT)

N = 134_217_728  # 512M fp32 = 2GiB per tensor
a = torch.randn(N)
b = torch.randn(N)
c = torch.empty(N)


def triad():
    torch.mul(a, 1.000001, out=c)
    torch.add(c, b, out=c)


triad()  # warm + page-in
t0 = time.time()
for _ in range(4):
    triad()
dt = (time.time() - t0) / 4
bytes_moved = N * 4 * 5  # triad 两步: 读a读b写c + 读c读b写c
print(f"MEM threads={NT}: triad {dt*1000:.0f} ms -> {bytes_moved/dt/1e9:.1f} GB/s", flush=True)

# 分配开销: 每次新分配 64MB 并 touch
t0 = time.time()
for _ in range(20):
    x = torch.empty(16 * 1024 * 1024)  # 64MB
    x.fill_(1.0)
dt = (time.time() - t0) / 20
print(f"MEM alloc+touch 64MB: {dt*1000:.1f} ms/次 "
      f"(若 >>5ms 则分配/页错误是瓶颈)", flush=True)

# 预分配复用对照
buf = torch.empty(16 * 1024 * 1024)
t0 = time.time()
for _ in range(20):
    buf.fill_(1.0)
dt = time.time() - t0
print(f"MEM reuse 64MB fill: {dt*1000/20:.1f} ms/次", flush=True)
