# -*- coding: utf-8 -*-
"""convert_dino_cache_f32.py - upcast feats.f16 -> feats.f32 (fp32 cache, resident RAM)."""
import os
import sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "data/dino_cache/v11_sym_full"
src = os.path.join(d, "feats.f16")
dst = os.path.join(d, "feats.f32")
if os.path.exists(dst):
    print("already exists:", dst)
    sys.exit(0)
mm = np.memmap(src, dtype=np.float16, mode="r")
n, pd = mm.size // (256 * 384), 256 * 384
print(f"upcasting {n} x 256 x 384 fp16 -> fp32 ...")
out = np.memmap(dst, dtype=np.float32, mode="w+", shape=(n, 256, 384))
B = 4096
for i in range(0, n, B):
    out[i:i + B] = mm[i * (256 * 384):(i + B) * (256 * 384)].reshape(-1, 256, 384).astype(np.float32)
out.flush()
del mm, out
print(f"done: {os.path.getsize(dst) / 2**30:.1f} GiB -> {dst}")
