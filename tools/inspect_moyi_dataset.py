import pandas as pd
import numpy as np
import os, glob

csv_path = "/home/ds/Workspace/moyi/assets/train_top10_style23_real.csv"
print(f"Reading {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("Sample rows:")
print(df[["img_id", "calligrapher", "script", "character", "calligrapher_id", "script_id", "character_id", "slot_name"]].head(5))

print("\nID ranges:")
print(f"  calligrapher_id: {df['calligrapher_id'].min()} ~ {df['calligrapher_id'].max()} (nunique={df['calligrapher_id'].nunique()})")
print(f"  script_id:       {df['script_id'].min()} ~ {df['script_id'].max()} (nunique={df['script_id'].nunique()})")
print(f"  character_id:    {df['character_id'].min()} ~ {df['character_id'].max()} (nunique={df['character_id'].nunique()})")
print(f"  img_id:          {df['img_id'].min()} ~ {df['img_id'].max()} (nunique={df['img_id'].nunique()})")

# Check shards
shards = {
    "shards_img": "/home/ds/Workspace/moyi/data/top10_style23/shards_img",
    "shards_std_w7": "/home/ds/Workspace/moyi/data/top10_style23/shards_std_w7",
    "shards_gtskel_w7": "/home/ds/Workspace/moyi/data/top10_style23/shards_gtskel_w7",
    "shards_aux_skel3": "/home/ds/Workspace/moyi/data/top10_style23/shards_aux_skel3",
}

for name, sdir in shards.items():
    files = sorted(glob.glob(os.path.join(sdir, "*.npz")))
    if not files:
        print(f"Shard {name}: EMPTY or NOT FOUND")
        continue
    total_imgs = 0
    with np.load(files[0]) as d:
        keys = list(d.keys())
        shape = d["latents"].shape
        dtype = d["latents"].dtype
        sample_ids = d["img_ids"][:5]
    for f in files:
        with np.load(f) as d:
            total_imgs += len(d["img_ids"])
    print(f"Shard {name}: {len(files)} npz files, {total_imgs} total images, shape per shard={shape}, keys={keys}, sample_ids={sample_ids}")
