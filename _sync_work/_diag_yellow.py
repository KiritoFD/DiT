# -*- coding: utf-8 -*-
"""_diag_yellow.py — 定位"黄底"来源.
假设: (a) VAE 对纯白 lat 的 decode 就偏黄; (b) 灰度图特有; (c) 数据里有黄底图.
"""
import os
import sys

import numpy as np
import torch as th
from PIL import Image
from torchvision import transforms as T

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SF = 0.18215
TF = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

from diffusers.models import AutoencoderKL  # noqa: E402
dev = "cuda"
vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema").to(dev).eval()

os.makedirs("_otout2", exist_ok=True)


def stats(tag, r, g, b, save=None):
    """r/g/b: numpy (H,W) float [0,255]"""
    mean = (r.mean(), g.mean(), b.mean())
    # 白底区域 = 亮度前 30%
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    m = lum >= np.percentile(lum, 70)
    bg = (r[m].mean(), g[m].mean(), b[m].mean())
    print(f"  [{tag}] 全图RGB={np.round(mean,1)} | 亮区RGB={np.round(bg,1)} "
          f"| R-B={bg[0]-bg[2]:+.1f}")
    if save:
        arr = np.stack([r, g, b], -1)
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(save)


def dec(lat):
    with th.no_grad(), th.autocast("cuda", dtype=th.bfloat16):
        d = vae.decode(lat / SF).sample
    a = ((d[0].float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(1, 2, 0) * 255
    return a[..., 0], a[..., 1], a[..., 2]


print("[1] 纯白图 (RGB 255,255,255) 的 encode->decode:")
white = TF(Image.new("RGB", (256, 256), (255, 255, 255))).unsqueeze(0).to(dev)
with th.no_grad():
    wl = (vae.encode(white).latent_dist.mode() * SF).float()
stats("white_roundtrip", *dec(wl), save="_otout2/white_rt.png")

print("\n[2] 纯灰图 (L=255 复制 3ch) —— 我们所有数据的处理方式:")
g = th.full((1, 3, 256, 256), 1.0).to(dev)      # (255/255-0.5)/0.5 = 1.0
with th.no_grad():
    gl = (vae.encode(g).latent_dist.mode() * SF).float()
stats("gray255_roundtrip", *dec(gl), save="_otout2/gray255_rt.png")

print("\n[3] latent 全 0 -> decode (wz 空间空白区的值):")
stats("zero_lat", *dec(th.zeros_like(wl)), save="_otout2/zero_rt.png")

print("\n[4] 真实骨架 latent 直接 decode（不用 VAE 重编, 读已有 shard）:")
import glob  # noqa: E402
sh = sorted(glob.glob("data/skel/aux_skel3_latents_base/shard_*.npz"))
if not sh:
    sh = sorted(glob.glob("data/skel/*/shard_*.npz"))
print(f"    用 shard: {sh[0] if sh else 'NONE'}")
if sh:
    z = np.load(sh[0])
    lat = th.from_numpy(z["latents"][:1]).float().to(dev)
    stats("skel_shard_raw", *dec(lat), save="_otout2/skel_shard_raw.png")
    wl_np = th.from_numpy(np.load("data/white_latent.npy")).float().to(dev)[None]
    stats("skel_shard_minus_white", *dec(lat - wl_np), save="_otout2/skel_shard_wz.png")

print("\n[5] img latent shard 对比:")
sh2 = sorted(glob.glob("data/latents/final_latents_base_shards/shard_*.npz")) or \
      sorted(glob.glob("data/latents/*/shard_*.npz"))
print(f"    用 shard: {sh2[0] if sh2 else 'NONE'}")
if sh2:
    z = np.load(sh2[0])
    lat = th.from_numpy(z["latents"][:1]).float().to(dev)
    stats("img_shard_raw", *dec(lat), save="_otout2/img_shard_raw.png")
