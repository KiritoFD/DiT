# -*- coding: utf-8 -*-
"""profile_cpu6.py — 决定性: 非连续内存布局对 mm/sdpa 的影响 + rope bmm 检查.

C1 mm: 连续 vs 转置视图 vs .contiguous() 后
C2 sdpa: 连续 (B,H,T,Dh) vs transpose 视图
C3 rope 实现路径 (若用 bmm/complex, 单独计时)
"""
import os, sys, time
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
import torch
torch.set_num_threads(NT)


def bench(f, n=20):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n


W = torch.randn(384, 1152)
big = torch.randn(384, 8192)
A_c = big.t()                      # 连续? t() 是视图 -> 非连续!
A_cc = big.t().contiguous()        # 真连续
print(f"A_nc.is_contiguous()={A_c.is_contiguous()}  A_cc.is_contiguous()={A_cc.is_contiguous()}")
dt_nc = bench(lambda: A_c @ W, 30)
dt_cc = bench(lambda: A_cc @ W, 30)
print(f"C1 mm: 非连续视图 {dt_nc*1000:.1f} ms | 连续 {dt_cc*1000:.1f} ms | {dt_nc/dt_cc:.1f}x", flush=True)

q = torch.randn(32, 256, 384)
qkv = torch.randn(32, 256, 1152)
qq, kk, vv = qkv.chunk(3, dim=-1)
qh_c = qq.view(32, 256, 6, 64).transpose(1, 2)          # 非连续视图
qh_cc = qh_c.contiguous()
kh_c = kk.view(32, 256, 6, 64).transpose(1, 2)
vh_c = vv.view(32, 256, 6, 64).transpose(1, 2)
with torch.no_grad():
    dt_nc = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qh_c, kh_c, vh_c), 30)
    dt_cc = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qh_cc, qh_cc, qh_cc), 30)
print(f"C2 sdpa: 非连续头视图 {dt_nc*1000:.1f} ms | 连续 {dt_cc*1000:.1f} ms | {dt_nc/dt_cc:.1f}x", flush=True)

# 模拟真实 block 输入: patch_embed 输出的 transpose 视图走 qkv linear
x_nc = big.t().view(32, 256, 384) if False else torch.randn(32, 32, 384).transpose(1, 2)  # (32,256,384) 非连续? transpose of (32,32,384)? 维度不同, 改成:
pe = torch.randn(32, 384, 256)           # conv 输出 (B, d, T)
x_view = pe.flatten(2).transpose(1, 2)   # (B, T, d) 非连续视图
lq = torch.nn.Linear(384, 1152)
with torch.no_grad():
    dt_nc = bench(lambda: lq(x_view), 30)
    dt_cc = bench(lambda: lq(x_view.contiguous()), 30)
print(f"C3 Linear(x_view): 非连续 {dt_nc*1000:.1f} ms | .contiguous()后 {dt_cc*1000:.1f} ms | {dt_nc/dt_cc:.1f}x", flush=True)

# rope 实现检查
import inspect
from src.model import modules as M
src = inspect.getsource(M)
print("\nC4 rope 相关源码片段:", flush=True)
import re
m = re.search(r"class RoPE.*?(?=\nclass |\Z)", src, re.S)
if m:
    print(m.group(0)[:1200], flush=True)
else:
    idx = src.find("rope")
    print(src[max(0, idx-200):idx+800], flush=True)
print("P6_DONE", flush=True)
