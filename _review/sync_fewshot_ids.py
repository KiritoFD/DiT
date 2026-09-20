"""把 fs_<cal>_all.csv 的 img_id 同步回 train/eval CSV，并让配置指向各自 shard。"""
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")
for c in ("怀素", "伊秉绶", "徐渭", "沈周"):
    allr = list(csv.DictReader(open(f"assets/fs_{c}_all.csv", encoding="utf-8")))
    idmap = {r["image_path"]: r["img_id"] for r in allr}
    for name in ("train", "eval"):
        p = f"assets/fs_{c}_{name}.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        cols = list(rows[0].keys())
        if "img_id" not in cols:
            cols.append("img_id")
        n = 0
        for r in rows:
            r["img_id"] = idmap.get(r["image_path"], "")
            if r["img_id"]:
                n += 1
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  {c}/{name}: {n}/{len(rows)} 有 img_id")
    cp = f"src/train/configs/v13_fs_{c}.json"
    d = json.load(open(cp, encoding="utf-8"))
    d["latent_shards_dir"] = f"data/50k/shards_fs_{c}"
    json.dump(d, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"    latent_shards_dir -> {d['latent_shards_dir']}")
