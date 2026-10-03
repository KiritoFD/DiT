# -*- coding: utf-8 -*-
"""检查 aux (canny/skel) latent 与 image latent 的数值分布, 验证"白底 bias 淹没"假设。"""
import glob
import os
import sys

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# 1) 白底 latent
from diffusers.models import AutoencoderKL
from torchvision import transforms as T
from PIL import Image

vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema").to("cpu").eval()
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
img = Image.new("RGB", (256, 256), "white")
x = tf(img).unsqueeze(0)
with th.no_grad():
    white = vae.encode(x).latent_dist.sample().mul_(0.18215)[0]
print("white latent: mean=%.4f std=%.4f min=%.3f max=%.3f"
      % (white.mean(), white.std(), white.min(), white.max()))
print("  per-channel mean:", np.round(white.mean(dim=(1, 2)).numpy(), 4))


def shard_stats(name, path):
    fps = sorted(glob.glob(os.path.join(path, "shard_*.npz")))
    if not fps:
        print(f"{name}: no shards"); return
    d = np.load(fps[0])
    lat = d["latents"][:200].astype(np.float32)
    print(f"\n{name}: shard={os.path.basename(fps[0])} n={lat.shape[0]} shape={lat.shape[1:]}")
    print("  mean=%.4f std=%.4f min=%.3f max=%.3f"
          % (lat.mean(), lat.std(), lat.min(), lat.max()))
    print("  per-channel mean:", np.round(lat.mean(axis=(0, 2, 3)), 4))
    # 与白底的接近程度: 每个空间位置到 white 的 L2 距离分布
    w = white.numpy()
    d2 = np.sqrt(((lat[:50] - w[None]) ** 2).sum(1))
    print("  dist-to-white: mean=%.3f p10=%.3f p50=%.3f p90=%.3f max=%.3f"
          % (d2.mean(), np.percentile(d2, 10), np.percentile(d2, 50),
             np.percentile(d2, 90), d2.max()))


shard_stats("IMG latent", "data/latents/final_latents_base_shards")
shard_stats("AUX skel3", "data/skel/aux_skel3_latents_base")
shard_stats("AUX canny", "data/aux/aux_canny_latents_base")
