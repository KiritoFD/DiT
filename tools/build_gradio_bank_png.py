"""从 data/50k/std/{glyph_id}.png 重新 VAE 编码构建 bank。

⚠ 为什么必须重做:
  data/50k/shards_std 里的 img_ids 不是 csv 行号（实测错位），
  之前用 int(v) -> rows[int(v)] 映射的 bank '小' 给出 '将' 字。
  这次直接从 PNG 编码，每个 glyph_id 唯一对应一张图，绝不错位。

bank 格式（gradio 期望）:
  keys:    script|char   （如 "楷|小"）
  latents: (N, 4, 32, 32)
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from diffusers.models import AutoencoderKL

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None
ROOT = "/root/Workspace/xy/DiT"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
STD = "data/50k/std"
CSV = "assets/train_50k_v2.csv"
OUT = "_sync_work/skel_bank_v13_png.npz"

# 1) csv: (script, char) -> glyph_id
import csv

gmap = {}
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    gmap[(r["script"], r["character"])] = r["glyph_id"]
print(f"  csv (script,char) -> glyph_id: {len(gmap)}")

# 2) VAE 编码所有 (script,char) 唯一对
unique = sorted(gmap.items())
print(f"  唯一 (script,char): {len(unique)}")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained(VAE).to(dev).eval()
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

def encode(item):
    (script, ch), gid = item
    p = os.path.join(STD, f"{gid}.png")
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("RGB")
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.LANCZOS)
    x = tf(im).unsqueeze(0).to(dev)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample() * 0.18215
    return (f"{script}|{ch}", z.cpu().float().numpy().astype(np.float16)[0])

t0 = __import__("time").time()
results = [None] * len(unique)
miss = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, item in enumerate(unique):
        r = encode(item)
        if r is None:
            miss += 1
            continue
        results[i] = r
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(unique)}  miss={miss}  "
                  f"elapsed={__import__('time').time()-t0:.0f}s", flush=True)
print(f"  编码完成: {sum(1 for r in results if r is not None)} 条, "
      f"miss={miss}, {__import__('time').time()-t0:.0f}s")

# 3) 写入 npz
keys = [r[0] for r in results if r is not None]
lats = np.stack([r[1] for r in results if r is not None])
np.savez(OUT, keys=np.array(keys), latents=lats)
print(f"  written {OUT}  ({lats.shape})")

# 4) 验证
for t in ("楷|小", "行|小", "楷|阜", "楷|将", "隶|阜"):
    print(f"   {t}: {'在' if t in keys else '不在'}")
