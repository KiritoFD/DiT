# -*- coding: utf-8 -*-
"""merge std skel shards: fame sym shards (img_id keyed) + expand shards (base rows) -> symlink dir."""
import glob
import os

OUT = "data/skel/std_skel3_latents_base_full"
os.makedirs(OUT, exist_ok=True)
n = 0
for src in glob.glob("data/skel/std_skel3_latents_fame_sym/shard_*.npz"):
    base = os.path.basename(src)
    dst = os.path.join(OUT, f"shard_f{base.split('_')[1]}")
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)
    n += 1
for src in glob.glob("data/skel/std_skel3_latents_base/expand_*.npz"):
    base = os.path.basename(src)
    dst = os.path.join(OUT, f"shard_e{base.split('_')[1]}")
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)
    n += 1
print(f"symlinked {n} shards -> {OUT}")
