"""隔离测 ZeroCrossAttention._inject: q_pos 到底改不改变它的输出。"""
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.dit import ZeroCrossAttention  # noqa: E402

torch.manual_seed(0)
a = ZeroCrossAttention(384, num_heads=6, grid_size=16, q_pos=False).eval()
b = ZeroCrossAttention(384, num_heads=6, grid_size=16, q_pos=True).eval()
# 同一份权重
b.load_state_dict(a.state_dict(), strict=False)
# 打破 out_proj 的 zero-init
with torch.no_grad():
    for m in (a, b):
        m.out_proj.weight.normal_(0, 0.02)
        m.out_proj.bias.normal_(0, 0.02)
with torch.no_grad():
    b.out_proj.weight.copy_(a.out_proj.weight)
    b.out_proj.bias.copy_(a.out_proj.bias)

x = torch.randn(2, 256, 384)
ctx = torch.randn(2, 256, 384)
with torch.no_grad():
    oa = a._inject(x, ctx)
    ob = b._inject(x, ctx)

print(f"  q_pos=False out_proj 是否零: {bool((a.out_proj.weight == 0).all())}")
print(f"  q_pos 分支条件 (b): N={x.shape[1]} <= ctx_pos={b.ctx_pos.shape[1]} -> "
      f"{x.shape[1] <= b.ctx_pos.shape[1]}")
d = (oa - ob).abs().max().item()
print(f"  _inject 输出差异: max|diff| = {d:.3e}")
print("  -> " + ("q_pos 确实改变 _inject 输出" if d > 1e-6
                 else "**q_pos 不改变 _inject 输出 —— 问题在模块内部**"))
