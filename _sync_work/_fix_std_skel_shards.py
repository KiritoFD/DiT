# -*- coding: utf-8 -*-
"""_fix_data/skel/std_skel_shards.py — 修复 std skel shards 的 img_ids（用图片 id，非 CSV 行号）。

build_data/skel/std_skel1_latents.py 的 bug: shards 存 img_ids=CSV行号,
而 latent_dataset 用图片 id(如 234327) 查询 → KeyError。
bank(data/skel/skel_bank_std1_v8.npz) 已含全部 latent, 这里只按 CSV 重新展开 shards。
"""
import os
import glob
import csv
import numpy as np

os.chdir("/root/Workspace/xy/DiT")

BANK = "data/skel/skel_bank_std1_v8.npz"
CSV = "assets/train_fame_clean_v8.csv"
OUT = "data/skel/std_skel1_latents_fame_v8"
SHARD_SIZE = 2592


def img_id_of(path):
    return int(os.path.splitext(os.path.basename(path))[0])


d = np.load(BANK, allow_pickle=True)
keys = [str(k) for k in d["keys"]]
lat = d["latents"]
print(f"bank: {lat.shape} keys={len(keys)}")
key2idx = {k: i for i, k in enumerate(keys)}

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.npz")):
    os.remove(f)

shard, ids = [], []
n_shard = 0
miss = 0


def flush():
    global shard, ids, n_shard
    if not shard:
        return
    np.savez(os.path.join(OUT, f"shard_{n_shard:05d}.npz"),
             latents=np.stack(shard).astype(np.float16),
             img_ids=np.array(ids, dtype=np.int64))
    n_shard += 1
    shard, ids = [], []


for r in rows:
    k = f'{r["script"]}|{r["character"]}'
    si = key2idx.get(k, -1)
    if si < 0:
        miss += 1
        continue
    shard.append(lat[si])
    ids.append(img_id_of(r["image_path"]))
    if len(shard) >= SHARD_SIZE:
        flush()
flush()

n_total = 0
for f in glob.glob(os.path.join(OUT, "*.npz")):
    dd = np.load(f)
    if "img_ids" in dd.files:
        n_total += len(dd["img_ids"])
    dd.close()
print("[done] shards=%d, total_samples=%d, miss=%d" % (n_shard, n_total, miss))
