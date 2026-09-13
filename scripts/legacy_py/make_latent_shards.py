# -*- coding: utf-8 -*-
"""在远程把官方 329715 张的 latent 组织成 shard（每 5000 张一个 npz）。
输入 final_latent_plan.json: {"img_id": {"src":"remote"/"local","path":...}}
输出 final_latents/shard_XXXXX.npz: {"latents": (N,4,32,32) f16, "img_ids": (N,) int32}
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

SHARD = 5000
plan = json.load(open("final_latent_plan.json", encoding="utf-8"))
print("total plans:", len(plan))

# 按 img_id 排序
ids = sorted(int(k) for k in plan)
print("img_id range:", ids[0], "-", ids[-1])

os.makedirs("final_latents", exist_ok=True)
batch_ids = []
batch_lat = []
n_shard = 0
n_loaded = 0
n_fail = 0

def load_lat(rec):
    p = rec["path"]
    if rec["src"] == "local":
        return np.load(p)   # latent_missing/{img_id}.npy
    else:
        return np.load(p)   # dataset/latents/....

for i, img_id in enumerate(ids):
    rec = plan[str(img_id)]
    try:
        lat = load_lat(rec)
        batch_ids.append(img_id)
        batch_lat.append(lat.astype(np.float16))
        n_loaded += 1
    except Exception as e:
        n_fail += 1
        if n_fail <= 5:
            print(f"  FAIL {img_id}: {e}", flush=True)
    if len(batch_ids) >= SHARD:
        arr = np.stack(batch_lat)
        ids_arr = np.array(batch_ids, dtype=np.int32)
        out = f"final_latents/shard_{n_shard:05d}.npz"
        np.savez(out, latents=arr, img_ids=ids_arr)
        n_shard += 1
        batch_ids, batch_lat = [], []
        if n_shard % 10 == 0:
            print(f"  {n_shard} shards, loaded {n_loaded}", flush=True)

if batch_ids:
    arr = np.stack(batch_lat)
    ids_arr = np.array(batch_ids, dtype=np.int32)
    out = f"final_latents/shard_{n_shard:05d}.npz"
    np.savez(out, latents=arr, img_ids=ids_arr)
    n_shard += 1

print(f"Done. shards={n_shard} loaded={n_loaded} fail={n_fail} total_ids={len(ids)}")
