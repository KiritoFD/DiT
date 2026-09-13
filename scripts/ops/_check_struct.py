#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 验证结构 loss 关键数值假设:
#   1) SD-VAE decode 输出像素值域 (SobelCannyLoss/SkeletonLoss 假设 [-1,1] -> (gray+1)/2)
#   2) canny/skel GT map 值域 ([0,1])
#   3) skel 的 ink 方向 (黑字白底 => ink=1-gray)
import os
import torch
import numpy as np

torch.manual_seed(0)
torch.set_num_threads(8)
os.chdir("/root/Workspace/xy/DiT")

# ---- 1) VAE decode 值域 (CPU, 避免与训练抢 GPU) ----
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema")
vae.eval()
vae = vae.to("cpu")
# 用真实 latent shard 抽几张来 decode (接近训练输入分布)
import glob
shard = sorted(glob.glob("final_latents/shard_*.npz"))[0]
d = np.load(shard)
lats = torch.from_numpy(d["latents"][:4]).float()  # (4,? ,4,32,32) maybe
print("latent shape:", lats.shape, "range", float(lats.min()), float(lats.max()))
if lats.dim() == 5:
    z = lats[:, 0]              # (B,4,32,32)
    z = z / z.shape[0] * 0 + z  # keep as-is (already per-image)
else:
    z = lats[:4]
z = z.detach()
z = z / 0.18215
with torch.no_grad():
    out = vae.decode(z).sample
print("VAE decode out shape:", out.shape, "min=%.4f max=%.4f mean=%.4f" % (out.min(), out.max(), out.mean()))
# 检查多少像素在[-1,1]外
frac_oob = ((out < -1.0) | (out > 1.0)).float().mean().item()
print("frac pixels outside [-1,1]: %.4f" % frac_oob)

# ---- 2) GT canny/skel 值域 ----
import PIL.Image as Image
c_dir = "final_canny"; s_dir = "final_skeleton"
for sub in ["final_canny", "final_skel" ]:
    pass
cf = os.path.join(c_dir, os.listdir(c_dir)[0])
sf = os.path.join(s_dir, os.listdir(s_dir)[0])
for name, p in [("canny", cf), ("skel", sf)]:
    im = np.asarray(Image.open(p).convert("L")).astype(np.float32)
    print(f"GT {name}: shape={im.shape} min={im.min():.2f} max={im.max():.2f} uniq={np.unique(im)[:6]}")

# ---- 3) 采样一个训练 batch 验证对齐 ----
from latent_dataset import MCCDLatentDataset
import pandas as pd
df = pd.read_csv("5script/train_top6.csv")
ds = MCCDLatentDataset(csv_file="5script/train_top6.csv", latent_shards_dir="final_latents",
                       img_root="final_imgs_256", canny_root=None, skel_root=None,
                       load_canny=True, load_skel=True)
b = ds[0]
print("sample keys:", list(b.keys()) if isinstance(b, dict) else type(b))
if isinstance(b, dict):
    for k, v in b.items():
        if isinstance(v, torch.Tensor):
            print("  ", k, tuple(v.shape), "range", float(v.min()), float(v.max()))
