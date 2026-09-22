"""测试 model_io.load_model_from_ckpt 能否加载各历史 ckpt。"""
import os
import sys

import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

from src.eval.model_io import load_model_from_ckpt  # noqa: E402

CKS = [
    ("v13base_155k", "assets/results/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt"),
    ("v13wd01_125k", "assets/results/v13_wd01/20260918-210256-v13-base-50k/checkpoints/0125000.pt"),
    ("v15c_fixed_210k", "assets/results/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt"),
    ("v15a_150k", "assets/results/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt"),
]

for name, p in CKS:
    if not os.path.exists(p):
        print(f"  {name}: ✗ ckpt 不存在")
        continue
    try:
        m, a = load_model_from_ckpt(p, device="cpu", use_ema=True, verbose=False)
        n = sum(x.numel() for x in m.parameters())
        print(f"  {name}: ✓ 加载成功, {n/1e6:.2f}M 参数")
    except Exception as e:
        print(f"  {name}: ✗ {type(e).__name__}: {str(e)[:150]}")
