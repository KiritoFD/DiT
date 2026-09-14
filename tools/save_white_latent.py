# -*- coding: utf-8 -*-
"""save_white_latent.py — 保存"白底图"的 VAE latent (推理时需加回, 见 rebuild_latents_wz)。

产物: data/white_latent.npy  (4,32,32) float32, 已是 *0.18215 之后的尺度。
"""
import os
import sys

import numpy as np
import torch as th
from PIL import Image
from torchvision import transforms as T

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from diffusers.models import AutoencoderKL       # noqa: E402

SF = 0.18215
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
vae = AutoencoderKL.from_pretrained(
    "data/pretrained/pretrained_models/sd-vae-ft-ema").cuda().eval()
x = tf(Image.new("RGB", (256, 256), "white")).unsqueeze(0).cuda()
with th.no_grad():
    wl = (vae.encode(x).latent_dist.mode() * SF)[0].float().cpu().numpy()
np.save("data/white_latent.npy", wl)
print("saved data/white_latent.npy", wl.shape,
      "per-ch mean:", np.round(wl.mean(axis=(1, 2)), 4))
