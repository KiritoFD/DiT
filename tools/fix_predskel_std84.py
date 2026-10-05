import os, sys, glob, csv, re
import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

csv_path = "assets/eval_top10_strict_subset84.csv"
shards_dir = "data/50k_v2_glyph15k/shards_std"
out_dir = "data/top10_style23/predskel_std84_e2e"

# Build index of official 50k_v2 shards_std
print("Building index of official 50k_v2 shards_std...")
shard_cache = {}
id_to_shard = {}
for p in sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz"))):
    with np.load(p) as z:
        for j, iid in enumerate(z["img_ids"]):
            id_to_shard[int(iid)] = (p, j)

def get_lat(iid):
    p, j = id_to_shard[iid]
    if p not in shard_cache:
        with np.load(p) as z:
            shard_cache[p] = np.array(z["latents"], copy=True)
    return shard_cache[p][j]

rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
print(f"Loaded {len(rows)} rows from {csv_path}")

latents = []
img_ids = []

for r in rows:
    iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
    old_id = int(r.get("old_50k_id", iid))
    
    # Try old_id first, then iid
    target_id = old_id if old_id in id_to_shard else iid
    assert target_id in id_to_shard, f"Missing id: {target_id}"
    
    lat = get_lat(target_id)
    latents.append(lat)
    img_ids.append(iid)

lat_arr = np.stack(latents).astype(np.float16)
id_arr = np.array(img_ids, dtype=np.int64)

# Also include seen20 ids from data/top10_style23/shards_std
seen_csv = "assets/eval_top10_seen_20.csv"
seen_shards = "data/top10_style23/shards_std"
seen_id_to_shard = {}
for p in sorted(glob.glob(os.path.join(seen_shards, "shard_*.npz"))):
    with np.load(p) as z:
        for j, iid in enumerate(z["img_ids"]):
            seen_id_to_shard[int(iid)] = (p, j)

seen_lats = []
seen_ids = []
for r in csv.DictReader(open(seen_csv, encoding="utf-8")):
    iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
    if iid not in img_ids:
        p, j = seen_id_to_shard[iid]
        with np.load(p) as z:
            seen_lats.append(z["latents"][j])
        seen_ids.append(iid)

if seen_ids:
    lat_arr = np.concatenate([lat_arr, np.stack(seen_lats).astype(np.float16)], axis=0)
    id_arr = np.concatenate([id_arr, np.array(seen_ids, dtype=np.int64)], axis=0)

os.makedirs(out_dir, exist_ok=True)
out_f = os.path.join(out_dir, "shard_00000.npz")
np.savez_compressed(out_f, latents=lat_arr, img_ids=id_arr)

print(f"✓ Successfully wrote clean, official shards to {out_f}!")
print(f"  Total samples: {len(id_arr)} (84 strict + {len(seen_ids)} seen)")
print(f"  Latents shape: {lat_arr.shape}, mean: {float(lat_arr.mean()):.4f}, std: {float(lat_arr.std()):.4f}")
