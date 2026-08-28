import os, csv, re, subprocess, json
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
print("scipy.ndimage OK")

# CPU count
print("nproc:", os.cpu_count())

# GPU info
import torch
print("GPU:", torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
print(f"VRAM free={free/2**30:.2f}G total={total/2**30:.2f}G")

# Max existing img id in train csv
csv_path = "5script/train_3top30_nobeike.csv"
max_id = 0
with open(csv_path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            max_id = max(max_id, int(m.group(1)))
print("max existing img id:", max_id)

# Latent shard count + shape
import glob
shards = sorted(glob.glob("final_latents/shard_*.npz"))
d = np.load(shards[0])
print("orig shards:", len(shards), "shape:", d["latents"].shape, d["latents"].dtype)
d.close()

# Disk free
s = os.statvfs("/root/Workspace/xy/DiT")
print(f"disk free: {s.f_bavail*s.f_frsize/2**30:.0f}G")