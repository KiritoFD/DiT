# -*- coding: utf-8 -*-
"""Verify mid-clean dataset: latent shards count/ids, csv rows, image coverage."""
import os, csv, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

total, mx, mn = 0, 0, 10**9
shards = sorted(glob.glob("final_latents_mid_clean/shard_*.npz"))
for p in shards:
    d = np.load(p)
    iids = d["img_ids"]
    total += len(iids)
    mx = max(mx, int(iids.max()))
    mn = min(mn, int(iids.min()))
    if len(iids) < 5000 or len(iids) > 5000:
        print(f"  shard {os.path.basename(p)}: {len(iids)} latents")
print(f"shards={len(shards)} total_latents={total} id_range=[{mn},{mx}]")

rows = list(csv.DictReader(open("5script/train_mid_clean.csv", encoding="utf-8")))
print(f"csv rows: {len(rows)}")
miss_img = sum(1 for r in rows if not os.path.exists(r["image_path"]))
print(f"missing images: {miss_img}")

# column sanity
print("cols:", list(rows[0].keys()))
# sample first + last
for r in (rows[0], rows[-1]):
    print("  sample:", r["image_path"], r["script"], r["character"], r["calligrapher"],
          r["calligrapher_id"], r["character_id"], r["glyph_id"], r["script_id"])

# distribution by script
from collections import Counter
c = Counter(r["script"] for r in rows)
print("by script:", dict(c))
print("OK")