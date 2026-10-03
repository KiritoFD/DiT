# -*- coding: utf-8 -*-
"""探查远程数据：train.csv 表头/行数、f4 分片格式、图片数量、img_id 关联"""
import os, sys, csv, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

print("CWD:", os.getcwd())
print("=" * 50)
print("1) train.csv")
with open("train.csv", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames
    print("  cols:", cols)
    n = 0
    sample = None
    for i, row in enumerate(rd):
        n += 1
        if sample is None:
            sample = dict(row)
    print("  rows:", n)
    print("  sample:", sample)

print("=" * 50)
print("2) data/latents/final_latents_f4 shards")
shards = sorted(glob.glob("data/latents/final_latents_f4/shard_*.npz"))
print("  n_shards:", len(shards))
d = np.load(shards[0])
print("  files:", d.files)
print("  latents:", d["latents"].shape, d["latents"].dtype)
print("  img_ids:", d["img_ids"].shape, d["img_ids"].dtype, d["img_ids"][:10])
total_ids = 0
for s in shards:
    total_ids += len(np.load(s)["img_ids"])
print("  total img_ids across shards:", total_ids)

print("=" * 50)
print("3) data/imgs/final_imgs_256")
pngs = glob.glob("data/imgs/final_imgs_256/*.png")
print("  n_png:", len(pngs))
if pngs:
    print("  sample:", sorted(pngs)[:5])

print("=" * 50)
print("4) img_id 交叉验证 (用 first 3 shards 的 id 查 train.csv 是否存在)")
ids = set()
for s in shards[:3]:
    ids.update(np.load(s)["img_ids"].tolist())
with open("train.csv", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    csv_ids = set()
    for row in rd:
        p = row.get("image_path", "")
        m = None
        import re
        m = re.search(r"(\d+)\.png", p)
        if m:
            csv_ids.add(int(m.group(1)))
        else:
            iid = row.get("img_id") or row.get("id")
            if iid:
                try:
                    csv_ids.add(int(iid))
                except Exception:
                    pass
print("  csv unique img ids:", len(csv_ids))
print("  shard ids in csv ids:", len(ids & csv_ids), "/", len(ids))
