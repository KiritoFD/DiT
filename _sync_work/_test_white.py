# -*- coding: utf-8 -*-
"""_test_white.py — 单测 maybe_add_white 对 4/8/12 通道都正确."""
import os
import sys

import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.inference import maybe_add_white, get_white_latent  # noqa: E402

print("white latent per-ch:", [round(float(x), 4)
                              for x in get_white_latent().mean(dim=(1, 2))])
for c in (4, 8, 12):
    x = torch.zeros(2, c, 32, 32)
    y = maybe_add_white(x, True)
    groups = [round(float(y[0, g * 4:(g + 1) * 4].mean()), 4) for g in range(c // 4)]
    print(f"C={c:2d} -> shape={tuple(y.shape)} 每组背景均值={groups}")
    assert tuple(y.shape) == (2, c, 32, 32), "shape mismatch"
    assert all(abs(v - float(get_white_latent().mean())) < 1e-3 for v in groups), "值不对"
x = torch.zeros(1, 4, 32, 32)
assert torch.equal(maybe_add_white(x, False), x), "zero_white=False 应原样返回"
print("ALL OK")
