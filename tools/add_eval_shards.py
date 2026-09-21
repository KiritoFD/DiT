"""把 eval 集的 std 也编码进 shards_std_fixed（原 shards_std 是训练+eval 合并的）。

原 data/50k/shards_std: 51,036 个 id（训练 50,786 + eval 250）
我重建时只做了训练的 50,786 -> eval 的 250 个全找不到 ✗
（strict 集是留出集，id 本来就不在训练集里）

做法: 把两个 eval csv 的 (old_50k_id, std_path) 也追加进 shards。
"""
import csv
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
OUT = "data/50k/shards_std_fixed"

# 现有 id
have = set()
for f in glob.glob(os.path.join(OUT, "*.npz")):
    have |= set(np.load(f)["img_ids"].tolist())
print(f"  现有 shards id: {len(have)}")

# 收集 eval 的 (old_50k_id, std_path)
pairs = []
for f in ("assets/eval_v13_strict_fixed.csv", "assets/eval_v13_seen_fixed.csv"):
    if not os.path.exists(f):
        continue
    for r in csv.DictReader(open(f, encoding="utf-8")):
        oid = r.get("old_50k_id", "").strip()
        if not oid:
            continue
        oid = int(oid)
        if oid in have:
            continue
        sp = r["std_path"]
        full = sp if os.path.isabs(sp) else os.path.join("/root/Workspace/xy/DiT", sp)
        pairs.append((oid, full))
print(f"  需要补的 eval 条目: {len(pairs)}")
if not pairs:
    print("  无需补充")
    raise SystemExit(0)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from diffusers.models import AutoencoderKL  # noqa: E402

vae = AutoencoderKL.from_pretrained(VAE).to(dev).eval()
tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# 追加为新 shard
nxt = len(glob.glob(os.path.join(OUT, "*.npz")))
B = 64
buf_lat, buf_id = [], []
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
    np.savez_compressed(os.path.join(OUT, f"shard_{nxt:05d}.npz"),
                        latents=np.concatenate(buf_lat, 0),
                        img_ids=np.asarray(buf_id, dtype=np.int64))
print(f"  ✓ shard_{nxt:05d}.npz ({len(buf_id)} 条)")

# 验证
have2 = set()
for f in glob.glob(os.path.join(OUT, "*.npz")):
    have2 |= set(np.load(f)["img_ids"].tolist())
print(f"  现在 shards id: {len(have2)}")
for f in ("assets/eval_v13_strict_fixed.csv", "assets/eval_v13_seen_fixed.csv"):
    if not os.path.exists(f):
        continue
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    v = set(int(r["old_50k_id"]) for r in rows if r["old_50k_id"].strip())
    print(f"  {os.path.basename(f)}: 缺 {len(v - have2)}/{len(v)}")
