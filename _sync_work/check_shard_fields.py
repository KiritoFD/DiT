import glob

import numpy as np

sp = sorted(glob.glob("data/50k/shards_std/shard_*.npz"))[0]
z = np.load(sp)
print("  file:", sp)
print("  files:", list(z.files))
for k in z.files:
    v = z[k]
    if isinstance(v, np.ndarray):
        print(f"   {k}: shape={v.shape} dtype={v.dtype} 样例={list(v[:3])}")
    else:
        print(f"   {k}: {type(v).__name__} = {str(v)[:80]}")
