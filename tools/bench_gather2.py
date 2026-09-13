# -*- coding: utf-8 -*-
"""bench_gather2.py - breakdown pinned fp32 gather cost."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils.dino_cache import DinoFeatureCache

cache = DinoFeatureCache("data/dino_cache/v11_sym_full")
print(cache.stats())
rng = np.random.default_rng(0)
table = cache._feats  # pinned fp32 (N,256,384)

# searchsorted only
t0 = time.time()
for _ in range(20):
    q = rng.choice(len(cache), 192, replace=False).astype(np.int64)
    pos = np.searchsorted(cache._sorted_ids, q)
print(f"searchsorted: {(time.time()-t0)/20*1000:.2f} ms")

# index_select only (CPU pinned -> pageable out)
rows_t = torch.from_numpy(rng.choice(len(cache), 192, replace=False))
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    out = torch.index_select(table, 0, rows_t)
print(f"index_select only: {(time.time()-t0)/20*1000:.1f} ms")

# index_select + H2D
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    out = torch.index_select(table, 0, rows_t).to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"index_select + H2D: {(time.time()-t0)/20*1000:.1f} ms")

# pinned rows: index_select on pinned rows tensor
rows_p = rows_t.pin_memory()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    out = torch.index_select(table, 0, rows_p).to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"index_select(pinned rows) + H2D: {(time.time()-t0)/20*1000:.1f} ms")

# direct fancy index on pinned table
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    out = table[rows_t].to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"table[rows] + H2D: {(time.time()-t0)/20*1000:.1f} ms")

# preallocated output + index_select(out=...)
out_buf = torch.empty(192, 256, 384, dtype=torch.float32, pin_memory=True)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    torch.index_select(table, 0, rows_t, out=out_buf)
    out_buf.to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"index_select(out=pinned) + H2D: {(time.time()-t0)/20*1000:.1f} ms")
