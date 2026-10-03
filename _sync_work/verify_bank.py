import collections

import numpy as np

z = np.load("_sync_work/skel_bank_std1_v8.npz", allow_pickle=True)
keys = [str(k) for k in z["keys"]]
lat = z["latents"]
print(f"  bank: {len(keys)} 条, latents {lat.shape}")
print(f"  key 样例: {keys[:4]}")
for t in ("楷|阜", "隶|阜", "行|阜"):
    mark = "在" if t in keys else "不在"
    print(f"   {t}: {mark}")
sc = collections.Counter(k.split("|")[0] for k in keys)
print(f"  按书体分布: {dict(sc)}")
n_chars = len(set(k.split("|")[1] for k in keys))
print(f"  去重字数: {n_chars}")
