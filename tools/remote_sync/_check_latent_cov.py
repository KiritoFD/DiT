# -*- coding: utf-8 -*-
"""验证: kailishu train csv 的子集 latent 是否都在 final_latents 缓存中。
输出缺失的样本数。"""
import csv, os, glob, numpy as np

rows = list(csv.DictReader(open("kailishu_train.csv", encoding="utf-8")))
# 解析 shard 里有哪些 img_id
shards = sorted(glob.glob("final_latents/shard_*.npz"))
available = set()
for sp in shards:
    d = np.load(sp)
    available.update(int(i) for i in d["img_ids"])
    d.close()
print(f"final_latents shards: {len(shards)}, 总 id: {len(available)}")
missing = 0
for r in rows:
    img_id = os.path.basename(r["image_path"])[:-4]
    if int(img_id) not in available:
        missing += 1
print(f"楷隶 {len(rows)} 行, latent 缺失 {missing}")
