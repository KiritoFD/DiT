"""重建 shards_std_fixed 里 044447 那条的 latent（std 从「升」改成了「陞」）。

## 为什么要重建
shards 是按 img_id 存的 VAE latent。std png 换了，latent 也得跟着换，
否则训练/评测时用的还是「升」的骨架。

## 做法
只重编码这一条，原地替换它所在的 shard 文件里的那一行（不重编全量，快）。
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

os.chdir("/root/Workspace/xy/DiT")

TARGET = 44447
SHARDS = "data/50k/shards_std_fixed"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
STD_PNG = "data/50k/std/044447.png"

# 1) 找到它在哪个 shard、第几行
loc = None
for f in sorted(glob.glob(os.path.join(SHARDS, "*.npz"))):
    with np.load(f) as d:
        ids = d["img_ids"]
        hit = np.where(ids == TARGET)[0]
        if len(hit):
            loc = (f, int(hit[0]), len(ids))
            break
if loc is None:
    raise SystemExit(f"  ✗ shards 里找不到 img_id={TARGET}")
f, row, total = loc
print(f"  找到: {os.path.basename(f)} 第 {row} 行（共 {total}）")

# 2) 重新编码
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from diffusers.models import AutoencoderKL  # noqa: E402

vae = AutoencoderKL.from_pretrained(VAE).to(dev).eval()
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)])
im = Image.open(STD_PNG).convert("RGB")
if im.size != (256, 256):
    im = im.resize((256, 256), Image.LANCZOS)
x = tf(im)[None].to(dev)
with torch.no_grad():
    lat = vae.encode(x).latent_dist.sample() * 0.18215
new_lat = lat.cpu().float().numpy().astype(np.float16)[0]
print(f"  新 latent: shape={new_lat.shape} std={new_lat.std():.4f}")

# 3) 原地替换
with np.load(f) as d:
    lats = d["latents"].copy()
    ids = d["img_ids"].copy()
old_std = float(lats[row].std())
lats[row] = new_lat
np.savez_compressed(f, latents=lats, img_ids=ids)
print(f"  替换: 旧 std={old_std:.4f} -> 新 std={float(lats[row].std()):.4f}")
print(f"  ✓ 已写回 {os.path.basename(f)}")

# 4) 复验
with np.load(f) as d:
    j = int(np.where(d["img_ids"] == TARGET)[0][0])
    print(f"  复验: img_id={d['img_ids'][j]} latent.std={d['latents'][j].std():.4f}")
