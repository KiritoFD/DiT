"""Correct coverage check: extract img_ids from top6/max and check against shard img_ids."""
import os, sys, csv, re
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

def load_ids(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in rows]
    return ids

shard_ids = set()
for fn in sorted(os.listdir("final_latents_f4")):
    if fn.endswith(".npz"):
        z = np.load(os.path.join("final_latents_f4", fn))
        shard_ids.update(int(i) for i in z["img_ids"])
        z.close()
print(f"shard img_ids: {len(shard_ids)}")

for csv_path, label in [("5script/train_top6.csv", "train_top6"),
                        ("5script/eval100_top6.csv", "eval100_top6"),
                        ("5script/show2_top6.csv", "show2_top6"),
                        ("5script/seen2_top6.csv", "seen2_top6")]:
    ids = load_ids(csv_path)
    missing = [i for i in ids if i not in shard_ids]
    print(f"{label}: n={len(ids)} missing_in_shards={len(missing)} {missing[:5]}")