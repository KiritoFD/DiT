# -*- coding: utf-8 -*-
"""Check ControlNet training-data readiness for a given csv:
1) CSV row count + unique img ids
2) latency coverage in latent shards dir 3) skel coverage in skel_root
"""
import os, sys, csv, re, glob, json
import numpy as np

csv_path = sys.argv[1]
latent_dir = sys.argv[2] if len(sys.argv) > 2 else "final_latents_mid_clean"
skel_root = sys.argv[3] if len(sys.argv) > 3 else "final_skeleton_d3"

ids = set()
n = 0
with open(csv_path, encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        m = re.search(r"(\d+)\.png", row["image_path"] or "")
        if not m:
            print("NO ID:", row.get("image_path"))
            continue
        ids.add(int(m.group(1)))
        n += 1
print(f"csv rows={n} unique_ids={len(ids)}")

# latent shards
lat_ids = set()
for p in sorted(glob.glob(os.path.join(latent_dir, "*.npz"))):
    z = np.load(p, mmap_mode="r")
    lat_ids.update(int(x) for x in z["img_ids"])
print(f"latent shards: {len(glob.glob(os.path.join(latent_dir, '*.npz')))} files, {len(lat_ids)} unique ids")
missing_lat = ids - lat_ids
print(f"rows missing latent: {len(missing_lat)}", sorted(missing_lat)[:10] if missing_lat else "")

# skel coverage (lazy: sample or count by listing files)
missing_skel = []
for i in ids:
    if not os.path.isfile(os.path.join(skel_root, f"{i}.png")):
        missing_skel.append(i)
pct = 100.0 * (len(ids) - len(missing_skel)) / max(len(ids), 1)
print(f"skel coverage: {pct:.2f}%  missing={len(missing_skel)}", 
      sorted(missing_skel)[:10] if missing_skel else "")