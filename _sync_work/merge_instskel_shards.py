# -*- coding: utf-8 -*-
"""把 build_skel_latents.py 产出的 794 个碎 shard 合并成**单个** npz。

为什么: build_latents 每攒满 --vae-batch(64) 就 flush 一次并立刻存盘, shard_size=5000
根本没机会生效 -> 50,786 张变成 794 个小文件。dataset preload 时要 np.load 794 次,
且 shard 目录里文件越多, `_skel_id_to_shard` 建表也越碎。合并成一个即可。
"""
import glob
import os
import shutil

import numpy as np

SRC = "data/50k/shards_instskel20"
TMP = "data/50k/_instskel20_merged"
ONE = "data/50k/_instskel20_one"

files = sorted(glob.glob(os.path.join(SRC, "shard_*.npz")))
print(f"[merge] {len(files)} shards from {SRC}")

lats, ids = [], []
for sp in files:
    with np.load(sp) as d:
        lats.append(d["latents"])
        ids.append(d["img_ids"])
L = np.concatenate(lats, 0)
I = np.concatenate(ids, 0)
print(f"[merge] total {L.shape} ids {I.shape} dtype={L.dtype}")

# 必须按 img_id 排序: dataset 用 img_id 建查表, 顺序乱不影响正确性但影响 shard 命名可读性
o = np.argsort(I)
L, I = L[o], I[o]
assert len(set(I.tolist())) == len(I), "img_id 有重复!"

os.makedirs(ONE, exist_ok=True)
out = os.path.join(ONE, "shard_00000.npz")
np.savez(out, latents=L, img_ids=I.astype(np.int64))
print(f"[merge] -> {out}  ({os.path.getsize(out)/1024**2:.0f} MB)")

# 原子替换: 先备份旧目录, 再换
shutil.rmtree(TMP, ignore_errors=True)
os.rename(SRC, TMP)
os.rename(ONE, SRC)
print(f"[merge] done. 旧碎目录备份在 {TMP} (确认无误后可删)")
