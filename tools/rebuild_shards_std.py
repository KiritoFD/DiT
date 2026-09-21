"""① 补那 30 行的 std_path（用渲染的新图）
   ② 重建 data/50k/shards_std（std_path 变了 1921 条，g 必须跟着换）
"""
import csv
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"

# ── ① 补 30 行的 std_path ────────────────────────────────────────────
added = json.load(open("assets/_std_added.json", encoding="utf-8"))
rows = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                                encoding="utf-8")))
cols = list(rows[0].keys())
n = 0
for r in rows:
    t = added.get(r["character"])
    sp = r["std_path"]
    full = sp if os.path.isabs(sp) else os.path.join("/root/Workspace/xy/DiT", sp)
    if t and not os.path.exists(full):
        r["std_path"] = t
        n += 1
with open("assets/train_50k_v2_fixed.csv", "w", newline="",
          encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"  ① 补了 {n} 行的 std_path（渲染的新骨架）")

# 校验所有 std_path 都存在
miss = 0
for r in rows:
    sp = r["std_path"]
    full = sp if os.path.isabs(sp) else os.path.join("/root/Workspace/xy/DiT", sp)
    if not os.path.exists(full):
        miss += 1
print(f"     std_path 缺失: {miss}/{len(rows)}")

# ── ② 重建 shards_std ────────────────────────────────────────────────
# ⚠⚠ img_id 必须是 **old_50k_id**（不是 std 文件编号！）
#   dataset 的 extract_img_id() 优先取 csv 的 img_id / old_50k_id 列，
#   所以 shard 查表用的是 old_50k_id。
#   实测: 原 shards_std 的 img_id 集合 == csv 的 old_50k_id 集合 ✓
#   （踩过: 用 std 文件编号 -> KeyError: 550）
OUT = "data/50k/shards_std_fixed"
os.makedirs(OUT, exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from diffusers.models import AutoencoderKL  # noqa: E402

vae = AutoencoderKL.from_pretrained(VAE).to(dev).eval()
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# (old_50k_id, std_path)
pairs = []
for r in rows:
    oid = r.get("old_50k_id", "").strip()
    if not oid:
        continue
    sp = r["std_path"]
    full = sp if os.path.isabs(sp) else os.path.join("/root/Workspace/xy/DiT", sp)
    pairs.append((int(oid), full))
print(f"  ② 条目: {len(pairs)}（按 old_50k_id）")

B = 64
buf_lat, buf_id, shard = [], [], 0
with torch.no_grad():
    for k in range(0, len(pairs), B):
        chunk = pairs[k:k + B]
        xs = []
        for _, f in chunk:
            im = Image.open(f).convert("RGB")
            if im.size != (256, 256):
                im = im.resize((256, 256), Image.LANCZOS)
            xs.append(tf(im))
        x = torch.stack(xs).to(dev)
        lat = vae.encode(x).latent_dist.sample() * 0.18215
        buf_lat.append(lat.cpu().float().numpy().astype(np.float16))
        buf_id.extend(i for i, _ in chunk)
        if len(buf_id) >= 5000:
            np.savez_compressed(os.path.join(OUT, f"shard_{shard:05d}.npz"),
                                latents=np.concatenate(buf_lat, 0),
                                img_ids=np.asarray(buf_id, dtype=np.int64))
            print(f"     shard_{shard:05d}.npz  ({len(buf_id)} 条)", flush=True)
            shard += 1
            buf_lat, buf_id = [], []
if buf_id:
    np.savez_compressed(os.path.join(OUT, f"shard_{shard:05d}.npz"),
                        latents=np.concatenate(buf_lat, 0),
                        img_ids=np.asarray(buf_id, dtype=np.int64))
    print(f"     shard_{shard:05d}.npz  ({len(buf_id)} 条)")
print(f"  ✓ {OUT} 重建完成")
