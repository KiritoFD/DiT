#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, torch
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval.in_mem_eval import _get_vae
from torchvision import transforms

vae = _get_vae("cpu")
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 1. 测量标准背景 latent z_bg
im_white = Image.new("RGB", (256, 256), (255, 255, 255))
with torch.no_grad():
    z_bg = vae.encode(tf(im_white)[None]).latent_dist.mean * 0.18215 # (1, 4, 32, 32)

# 2. 测量墨迹 delta_z
im_black = Image.new("RGB", (256, 256), (0, 0, 0))
with torch.no_grad():
    z_ink = vae.encode(tf(im_black)[None]).latent_dist.mean * 0.18215
delta_ink = (z_ink - z_bg).mean(dim=(2, 3), keepdim=True) # (1, 4, 1, 1)

# 3. 载入真实骨架 '出'
p_std = "data/50k/std/002155.png"
with torch.no_grad():
    g = vae.encode(tf(Image.open(p_std).convert("RGB"))[None]).latent_dist.mean * 0.18215

# 测试剪刀: 在中间横向切一刀 (抹去中间的一段笔画)
mask_prune = torch.zeros(1, 1, 32, 32)
mask_prune[:, :, 14:18, :] = 1.0 # 中间 4 行完全抹除

# 方案 A: 错误的直接乘以 (1 - mask) -> 变成 0
g_bad = g * (1.0 - mask_prune)

# 方案 B: 正确的向 z_bg 插值
g_good = (1.0 - mask_prune) * g + mask_prune * z_bg

# 测试胶水: 在底部画一条水平连接线
mask_lig = torch.zeros(1, 1, 32, 32)
mask_lig[:, :, 28:30, 8:24] = 1.0 # 底部横线
g_lig = g_good + mask_lig * delta_ink

# 解码所有
with torch.no_grad():
    dec_orig = vae.decode(g / 0.18215).sample
    dec_bad = vae.decode(g_bad / 0.18215).sample
    dec_good = vae.decode(g_good / 0.18215).sample
    dec_lig = vae.decode(g_lig / 0.18215).sample

for name, dec in [("原始骨架", dec_orig), ("错误置零剪刀(灰黄棕伪影)", dec_bad), ("正确白底插值剪刀(干净抹除)", dec_good), ("正确胶水(增加连接线)", dec_lig)]:
    arr = ((dec.float().clamp(-1, 1) + 1) / 2).cpu().numpy()[0, 0]
    sub = np.zeros((32, 32))
    for y in range(32):
        for x in range(32):
            sub[y, x] = arr[y*8:(y+1)*8, x*8:(x+1)*8].min()
    print(f"\n=== {name} (min={arr.min():.2f}, mean={arr.mean():.2f}) ===")
    for y in range(10, 24):
        line = "".join(["#" if sub[y, x] < 0.8 else " " for x in range(32)])
        if "#" in line:
            print(f"{y:>2}: {line}")
