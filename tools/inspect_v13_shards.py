import os
import numpy as np
import torch
import csv
from PIL import Image
from diffusers.models import AutoencoderKL

base = "/root/Workspace/xy/DiT"

# 1. 检查 data/50k/shards_std 里的内容
p_std = os.path.join(base, "data/50k/shards_std/shard_00000.npz")
if os.path.exists(p_std):
    data = np.load(p_std)
    keys = list(data.keys())
    print(f"shard_00000.npz: {len(keys)} entries, sample keys: {keys[:5]}")
    sample_key = keys[0]
    lat = data[sample_key]
    print(f"Sample latent shape: {lat.shape}, dtype={lat.dtype}, mean={lat.mean():.4f}, std={lat.std():.4f}")

# 2. 检查 data/50k/shards_aux_skel3 里的内容 (对比)
p_aux = os.path.join(base, "data/50k/shards_aux_skel3/shard_00000.npz")
if os.path.exists(p_aux):
    data_aux = np.load(p_aux)
    keys_aux = list(data_aux.keys())
    print(f"shards_aux_skel3: {len(keys_aux)} entries")
    lat_aux = data_aux[sample_key] if sample_key in data_aux else None
    if lat_aux is not None:
        print(f"Aux latent shape: {lat_aux.shape}, mean={lat_aux.mean():.4f}, diff vs std={(lat - lat_aux).mean():.4f}")

# 3. 检查 train_50k_v2.csv 里 sample_key 对应什么图
train_csv = os.path.join(base, "assets/train_50k_v2.csv")
if os.path.exists(train_csv):
    rows = list(csv.DictReader(open(train_csv, encoding="utf-8")))[:5]
    print("train_50k_v2.csv sample rows:")
    for r in rows:
        print(f"  img_id={r.get('img_id')}, char={r.get('char') or r.get('character')}, callig={r.get('calligrapher')}, std_path={r.get('std_path')}")
