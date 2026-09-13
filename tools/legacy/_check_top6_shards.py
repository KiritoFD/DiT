"""Quick check: do top6 train paths exist in final_latents_f4 shards?"""
import os, sys, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

rows = list(csv.DictReader(open("5script/train_top6.csv", encoding="utf-8")))
paths = [r["image_path"] for r in rows]
print(f"top6 paths: {len(paths)}")

shard_dir = "final_latents_f4"
keys = set()
for fn in sorted(os.listdir(shard_dir)):
    if fn.endswith(".npz"):
        try:
            z = np.load(os.path.join(shard_dir, fn), allow_pickle=True)
            keys.update(z.files)
            z.close()
        except Exception as e:
            print(f"  skip {fn}: {e}")
print(f"total keys in shards: {len(keys)}")
missing = [p for p in paths if p not in keys]
print(f"top6 missing from shards: {len(missing)}")
if missing[:5]:
    for m in missing[:5]:
        print(f"  missing: {m}")
# show a sample key format
for k in list(keys)[:3]:
    print(f"  sample key: {k!r}")