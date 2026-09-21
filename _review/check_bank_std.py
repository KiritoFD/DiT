import collections

import numpy as np

z = np.load("data/skel/skel_bank_std.npz", allow_pickle=True)
k = z["keys"]
L = z["latents"]
print(f"  keys: {len(k)}, latents: {L.shape}")
print(f"  前5: {k[:5]}")
print(f"  楷|阜 在: {'楷|阜' in k}")
print(f"  行|阜 在: {'行|阜' in k}")
print(f"  隶|阜 在: {'隶|阜' in k}")
c = collections.Counter(x.split("|")[0] for x in k)
print(f"  按书体: {dict(c)}")
print(f"  去重字数: {len(set(x.split('|')[1] for x in k))}")

# 对比 wd01 用 7500+ 模的 bank 来生成「小」楷 于右任
# 看 '小' 在 keys 里吗
for t in ("楷|小", "行|小", "楷|将", "楷|阜"):
    print(f"   {t}: 在={'{t}' in [str(x) for x in k]}")