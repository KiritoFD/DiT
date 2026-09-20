#!/usr/bin/env python
"""为 few-shot 的新书家图**预编码 latent**，生成训练/评测用的 shard。

⚠ 为什么必须做: 训练走 **latent-only 模式**（`[infra] VAE skipped`），
   数据加载器从 shard 读 latent，不读图。wild 的图没有 latent，
   直接喂 CSV 会报 `无法提取 img_id (preload)`。

产物: <shards>/shard_00000.npz  {latents:(N,4,32,32) f16, img_ids:(N,)}
     并把 img_id 写回 CSV（新增/覆盖 img_id 列）

用法:
  python tools/encode_fewshot_latents.py \
      --csv assets/fs_怀素_train.csv --out data/50k/shards_fs_怀素
  python tools/encode_fewshot_latents.py \
      --csv assets/fs_怀素_eval.csv  --out data/50k/shards_fs_怀素_eval
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
Image.MAX_IMAGE_PIXELS = None

VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
# ⚠ 别写成 data/pretrained/sd-vae-ft-ema —— 那个路径不存在（实测踩过）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="shard 输出目录")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"[enc] {a.csv}: {len(rows)} 行")

    # img_id = 行序号（shard 里用 str 对齐）
    paths = []
    for i, r in enumerate(rows):
        p = r["image_path"]
        if not os.path.isabs(p):
            p = os.path.join(ROOT, p)
        paths.append(p)
    miss = sum(1 for p in paths if not os.path.exists(p))
    if miss:
        print(f"  ⚠ {miss} 张图不存在")
    idx = [i for i, p in enumerate(paths) if os.path.exists(p)]
    print(f"  可读 {len(idx)} / {len(paths)}")

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(VAE).to(dev).eval()
    tf = transforms.Compose([
        transforms.Resize((a.size, a.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    lats, ids = [], []
    with torch.no_grad():
        for s in range(0, len(idx), a.batch):
            bi = idx[s:s + a.batch]
            imgs = []
            for i in bi:
                im = Image.open(paths[i]).convert("RGB")
                imgs.append(tf(im))
            x = torch.stack(imgs).to(dev)
            z = vae.encode(x).latent_dist.sample()
            z = z * 0.18215
            lats.append(z.float().cpu().numpy().astype(np.float16))
            ids.extend([str(i) for i in bi])
            if s % (a.batch * 10) == 0:
                print(f"    {s}/{len(idx)}")
    lat = np.concatenate(lats, 0)
    print(f"  latents {lat.shape} {lat.dtype}")

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "shard_00000.npz"),
             latents=lat, img_ids=np.array(ids, dtype="<U32"))
    print(f"  written {a.out}/shard_00000.npz")

    # 写回 img_id
    cols = list(rows[0].keys())
    if "img_id" not in cols:
        cols.append("img_id")
    for i, r in enumerate(rows):
        r["img_id"] = str(i)
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  img_id 已写回 {a.csv}")


if __name__ == "__main__":
    main()
