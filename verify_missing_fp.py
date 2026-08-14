# -*- coding: utf-8 -*-
"""用本地 12414 个 latent 算 fp，与远程 latent_fp 集合比对，验证缺失判断。"""
import os, sys, json, glob, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

# 加载远程 fp 集合
rem = json.load(open("_remote_vae_index.json", encoding="utf-8"))
rem_fp = set(r["latent_fp"] for r in rem)
print("remote fp count:", len(rem_fp))

# 本地缺失 latent
files = sorted(glob.glob("latent_missing/*.npy"))
print("local missing latent files:", len(files))

def latent_fp(lat_f16):
    arr = (lat_f16.astype(np.float64) * 1000).round().astype(np.int16)
    return hashlib.md5(arr.tobytes()).hexdigest()

match = 0
no_match = 0
match_ids = []
for f in files:
    lat = np.load(f)
    fp = latent_fp(lat)
    if fp in rem_fp:
        match += 1
        match_ids.append(os.path.basename(f)[:-4])
    else:
        no_match += 1

print(f"match in remote: {match}, no-match: {no_match}")
if match_ids:
    print("matched img_ids sample:", match_ids[:20])
