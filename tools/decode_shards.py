import os
import numpy as np
import torch
from PIL import Image
from diffusers.models import AutoencoderKL

base = "/root/Workspace/xy/DiT"
vae_path = os.path.join(base, "pretrained_models/sd-vae-ft-ema")
vae = AutoencoderKL.from_pretrained(vae_path).to("cpu").eval()
sf = 0.18215

# 1. 解码 data/50k/shards_std 中的前 2 个样本
p_std = os.path.join(base, "data/50k/shards_std/shard_00000.npz")
d_std = np.load(p_std)
lats_std = d_std["latents"][:2]
ids_std = d_std["img_ids"][:2]

for i in range(2):
    lat = torch.from_numpy(lats_std[i:i+1]).float()
    with torch.no_grad():
        dec = vae.decode(lat / sf).sample
    arr = dec[0, :3].permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().numpy()
    Image.fromarray(arr).save(f"/tmp/decoded_std_{ids_std[i]}.png")
    print(f"Decoded std sample {ids_std[i]} to /tmp/decoded_std_{ids_std[i]}.png")

# 2. 解码 data/50k/shards_aux_skel3 中的前 2 个样本
p_aux = os.path.join(base, "data/50k/shards_aux_skel3/shard_00000.npz")
d_aux = np.load(p_aux)
lats_aux = d_aux["latents"][:2]
ids_aux = d_aux["img_ids"][:2]

for i in range(2):
    lat = torch.from_numpy(lats_aux[i:i+1]).float()
    with torch.no_grad():
        dec = vae.decode(lat / sf).sample
    arr = dec[0, :3].permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().numpy()
    Image.fromarray(arr).save(f"/tmp/decoded_aux_{ids_aux[i]}.png")
    print(f"Decoded aux sample {ids_aux[i]} to /tmp/decoded_aux_{ids_aux[i]}.png")
