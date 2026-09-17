# -*- coding: utf-8 -*-
"""12ch 前提检查: px60 train csv 的 img_id 在现有 aux shards 里的覆盖率。"""
import csv
import glob
import os

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
ids = set()
with open(os.path.join(ROOT, "assets/train_fame-kxl-tj-px60.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        import re
        ids.add(int(re.search(r"(\d+)\.png", row["image_path"]).group(1)))
print(f"px60 train ids: {len(ids)}")

for d in ["aux_canny_latents_fame_e", "aux_skel_latents_fame_e",
          "aux_canny_latents_v8", "aux_skel3_latents_v8",
          "aux_canny_latents_sym", "aux_skel3_latents_sym",
          "aux_canny_latents_base", "aux_skel3_latents_v8"]:
    p = os.path.join(ROOT, "data/aux", d)
    if not os.path.isdir(p):
        print(f"{d:<28} (no dir)")
        continue
    cov = set()
    ch = None
    for sp in sorted(glob.glob(os.path.join(p, "shard_*.npz"))):
        with np.load(sp) as z:
            cov.update(int(x) for x in z["img_ids"])
            if ch is None:
                ch = z["latents"].shape[1:]
    n_hit = len(ids & cov)
    print(f"{d:<28} n={len(cov):>7}  hit {n_hit}/{len(ids)} = {n_hit/len(ids)*100:5.1f}%  shape={ch}")
