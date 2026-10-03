import sys, os
ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch as th
import numpy as np
from PIL import Image

d = np.load("data/top10_style23/shards_img/shard_00000.npz")
lat = th.from_numpy(d["latents"][:4]).float()

from src.eval.in_mem_eval import _get_vae
dev = th.device("cuda")
vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()

with th.no_grad():
    dec = (vae.decode(lat.to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2

for i in range(4):
    im = Image.fromarray((dec[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
    im.save(f"test_shards_img_{i}.png")
    arr = np.array(im)
    print(f"Sample {i}: shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
