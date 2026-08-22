#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把小网络训练数据打包: latent(4,32,32,f16) + skel(256,256) + canny(256,256)。
   输出 5script/structnet/ 下三个 memmap npy, 供 CPU 训练随机访问。
   latent 值域: 使用 shard 原始 float16 (与 latent_dataset 一致, 不改缩放)。
   规模: top6 train ~10866 张; 这里直接用 5script/train_top6.csv 对应用图。
"""
import os, sys, csv, re, time
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
BASE = "/root/Workspace/xy/DiT/"
CSV = BASE + "5script/train.csv"
SHARDS = BASE + "final_latents"
CAN_ROOT = BASE + "final_canny"
SKEL_ROOT = BASE + "final_skeleton"
OUT = BASE + "5script/structnet"
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in rows]
n = len(rows)
print(f"rows={n}", flush=True)

# ---- 建立 shard 索引: img_id -> (shard_path, offset) ----
id_to_loc = {}
for sp in sorted(os.listdir(SHARDS)):
    if not sp.endswith(".npz"):
        continue
    full = os.path.join(SHARDS, sp)
    d = np.load(full)
    for j, iid in enumerate(d["img_ids"]):
        id_to_loc[int(iid)] = (full, j)
    d.close()
print(f"indexed {len(id_to_loc)} latents", flush=True)

# ---- latent memmap ----
lt_path = OUT + "/latents.npy"
if not os.path.exists(lt_path):
    t0 = time.time()
    arr = np.lib.format.open_memmap(lt_path, mode="w+", dtype=np.float16, shape=(n, 4, 32, 32))
    miss = 0
    for i, iid in enumerate(ids):
        loc = id_to_loc.get(iid)
        if loc is None:
            miss += 1
            continue
        with np.load(loc[0]) as sh:
            arr[i] = sh["latents"][loc[1]]
        if (i + 1) % 3000 == 0:
            print(f"  lat {i+1}/{n} {time.time()-t0:.0f}s miss={miss}", flush=True)
    arr.flush(); del arr
    print(f"latents.npy written ({time.time()-t0:.0f}s, miss={miss})", flush=True)
else:
    print("skip latents.npy (exists)", flush=True)

def pack_gray(name, root, fn):
    p = OUT + f"/{name}.npy"
    if os.path.exists(p):
        print(f"skip {p} (exists)", flush=True)
        return
    t0 = time.time()
    arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(n, 256, 256))
    for i, iid in enumerate(ids):
        with Image.open(os.path.join(root, f"{iid}.png")) as im:
            arr[i] = np.asarray(im.convert("L"))
        if (i + 1) % 3000 == 0:
            print(f"  {name} {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    arr.flush(); del arr
    print(f"{name}.npy written ({time.time()-t0:.0f}s)", flush=True)

pack_gray("skels", SKEL_ROOT, None)
pack_gray("cannys", CAN_ROOT, None)

with open(OUT + "/ids.txt", "w") as f:
    f.write("\n".join(str(i) for i in ids))
print("ALL DONE", flush=True)