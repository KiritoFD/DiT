import sys, os
ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import glob
import pandas as pd

df = pd.read_csv("assets/train_top10_style23_real.csv")
print(f"train_top10_style23_real.csv total rows: {len(df)}")

shards_to_check = [
    "data/top10_style23/shards_img",
    "data/top10_style23/shards_gtskel_w7",
    "data/top10_style23/shards_aux_skel3",
    "data/top10_style23/shards_predskel_v31",
]

for s in shards_to_check:
    flist = sorted(glob.glob(f"{s}/*.npz"))
    if flist:
        d = np.load(flist[0])
        print(f"[{s}] shards: {len(flist)}, shard_00000 keys: {len(d.files)}, shape: {d[d.files[0]].shape}")
    else:
        print(f"[{s}] NOT FOUND or EMPTY!")
