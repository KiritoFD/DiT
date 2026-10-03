import numpy as np
d = np.load("data/top10_style23/shards_gtskel_w7/shard_00000.npz")
print("Keys in shards_gtskel_w7:", list(d.keys()))
for k in d.keys():
    print(f"  {k}: shape {d[k].shape}, dtype {d[k].dtype}")
