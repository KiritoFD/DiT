# -*- coding: utf-8 -*-
"""Fast id/skel coverage check: avoid os.listdir on huge dirs; use glob on latents only."""
import os, sys, glob, re, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# max id in latent shards
lat_dir = os.path.join(BASE, "final_latents")
max_lat_id = 0
n_lat = 0
for sp in sorted(glob.glob(os.path.join(lat_dir, "shard_*.npz"))):
    d = np.load(sp)
    iids = d["img_ids"]
    n_lat += len(iids)
    max_lat_id = max(max_lat_id, int(iids.max()))
    d.close()
print(f"final_latents: {n_lat} latents, max id {max_lat_id}", flush=True)

# clean csv ids + check a few images exist by direct path (not listdir)
rows = list(csv.DictReader(open(os.path.join(BASE, "5script", "train_3top30_common.csv"), encoding="utf-8")))
clean_ids = set()
for r in rows:
    m = re.search(r"(\d+)\.png", r["image_path"])
    if m:
        clean_ids.add(int(m.group(1)))
print(f"clean csv: {len(clean_ids)} unique img ids, range [{min(clean_ids)}, {max(clean_ids)}]", flush=True)

# sample 200 clean ids, check image + skeleton exist
import random
random.seed(0)
sample = random.sample(sorted(clean_ids), min(200, len(clean_ids)))
img_ok = sum(os.path.exists(os.path.join(BASE, "final_imgs_256", f"{i}.png")) for i in sample)
sk_ok = sum(os.path.exists(os.path.join(BASE, "final_skeleton_d3", f"{i}.png")) for i in sample)
print(f"sampled 200 ids: img present {img_ok}/200, skel present {sk_ok}/200", flush=True)
# check no existing id >= 1000000 among latents (collision with NEXT_ID)
print(f"NEXT_ID=1000000 > max_lat_id={max_lat_id}: {1000000 > max_lat_id}", flush=True)
print("DONE", flush=True)
