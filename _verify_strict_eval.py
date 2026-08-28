import csv, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

def ids(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s.add(int(r["image_path"].rsplit("/", 1)[1].replace(".png", "")))
    return s

ev = ids("5script/eval_strict_top6.csv")
print("eval_strict_top6 rows:", len(ev))

lat_ids = set()
for sp in sorted(glob.glob("final_latents/shard_*.npz")):
    d = np.load(sp)
    lat_ids.update(int(x) for x in d["img_ids"])
    d.close()
miss = ev - lat_ids
print("missing from final_latents:", len(miss), sorted(miss)[:10])

# png presence
import os
miss_png = [i for i in ev if not os.path.exists(f"final_imgs_256/{i}.png")]
print("missing png:", len(miss_png))