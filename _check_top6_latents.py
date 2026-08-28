import csv, json, os, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

def ids_from_csv(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iid = int(r["image_path"].rsplit("/", 1)[1].replace(".png", ""))
            ids.add(iid)
    return ids

train6 = ids_from_csv("5script/train_top6.csv")
eval100_6 = ids_from_csv("5script/eval100_top6.csv")
print("train_top6 imgs:", len(train6))
print("eval100_top6 imgs:", len(eval100_6))
print("overlap train/eval100:", len(train6 & eval100_6))

# scan final_latents shards
lat_ids = set()
for sp in sorted(glob.glob("final_latents/shard_*.npz")):
    d = np.load(sp)
    lat_ids.update(int(x) for x in d["img_ids"])
    d.close()
print("final_latents ids:", len(lat_ids), "max:", max(lat_ids) if lat_ids else None)

miss_train = train6 - lat_ids
miss_eval = eval100_6 - lat_ids
print("train_top6 ids NOT in final_latents:", len(miss_train))
print("eval100_top6 ids NOT in final_latents:", len(miss_eval))
if miss_eval:
    print("  sample missing eval ids:", sorted(miss_eval)[:10])

# check imgs on disk
missing_png = [i for i in list(train6)[:2000] if not os.path.exists(f"final_imgs_256/{i}.png")]
print("train_top6 imgs missing png (first2000):", len(missing_png))
missing_eval_png = [i for i in eval100_6 if not os.path.exists(f"final_imgs_256/{i}.png")]
print("eval100_top6 imgs missing png:", len(missing_eval_png))

# manifest check
man_path = "archive/final_manifest.json"
if os.path.exists(man_path):
    man = json.load(open(man_path, encoding="utf-8"))
    print("manifest entries:", len(man))
    # does manifest cover eval100_top6 ids?
    mid2e = {e["img_id"]: e for e in man}
    covered = sum(1 for i in eval100_6 if i in mid2e)
    print("eval100_top6 ids in manifest:", covered, "/", len(eval100_6))
    # sample manifest keys
    if man:
        print("  sample manifest keys:", list(man[0].keys()))
else:
    print("no local archive/final_manifest.json")