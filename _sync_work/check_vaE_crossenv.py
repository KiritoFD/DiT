#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查跨环境 VAE 编码是否一致：同一张 std png，两个解释器分别编码，比较。

动机: 离线增强 shards 是 /opt/conda/bin/python (torch 1.13.1) 编的,
     而训练/基础 shards 用 cu121 (torch 2.5.1)。若跨环境有系统差,
      那"几何扰动"里会混进环境伪影。
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

os.chdir('/root/Workspace/xy/DiT')
from diffusers.models import AutoencoderKL  # noqa: E402

print(f'torch {torch.__version__}', flush=True)
vae = AutoencoderKL.from_pretrained(
    'data/pretrained/pretrained_models/sd-vae-ft-ema', local_files_only=True).eval()
tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])

ids = [0, 1, 100, 550, 9999]
outs = {}
for i in ids:
    p = f'data/50k/std/{i:06d}.png'
    if not os.path.exists(p):
        continue
    x = tf(Image.open(p).convert('RGB'))[None]
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mode() * 0.18215
    outs[i] = lat.numpy().astype(np.float32)
np.savez(sys.argv[1], **{str(k): v for k, v in outs.items()})
print(f'saved {sys.argv[1]}: {sorted(outs)}', flush=True)
