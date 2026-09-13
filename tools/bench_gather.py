# -*- coding: utf-8 -*-
"""bench_gather.py - measure DinoFeatureCache.gather cost per training step."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.dino_cache import DinoFeatureCache

print("free RAM check:")
os.system("free -g | head -2")

cache = DinoFeatureCache("data/dino_cache/v11_sym_full")
print(cache.stats())
rng = np.random.default_rng(0)
ids_all = cache._ids

# warm page cache
t0 = time.time()
cache.gather(ids_all[:192], device="cuda")
torch.cuda.synchronize()
print(f"warm gather: {(time.time()-t0)*1000:.1f} ms")

# steady-state: 20 batches like training
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    idx = rng.choice(len(cache), 192, replace=False)
    feats, missing = cache.gather(torch.from_numpy(ids_all[idx]), device="cuda")
torch.cuda.synchronize()
dt = (time.time() - t0) / 20 * 1000
print(f"steady gather: {dt:.1f} ms/step (batch=192) -> {dt/268*100:.0f}% of a 268ms step")

# breakdown: numpy fancy index alone
t0 = time.time()
for _ in range(20):
    idx = rng.choice(len(cache), 192, replace=False)
    _ = cache._feats[torch.from_numpy(idx).numpy()]
print(f"numpy fancy-index only: {(time.time()-t0)/20*1000:.1f} ms")

# H2D transfer alone (fp16 vs fp32)
t = torch.from_numpy(np.ascontiguousarray(cache._feats[:192]))
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    _ = t.to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"H2D fp16 37MB: {(time.time()-t0)/20*1000:.1f} ms")
t32 = t.float()
t0 = time.time()
for _ in range(20):
    _ = t32.to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"H2D fp32 75MB: {(time.time()-t0)/20*1000:.1f} ms")
