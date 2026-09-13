# -*- coding: utf-8 -*-
"""build_image_latents.py — 通用图像 VAE latent shards 构建 (任意 csv).

对 csv 每行 image_path: 灰度 -> 256 -> VAE encode -> (4,32,32) latent (x0.18215),
按 shard 保存 {latents fp16, img_ids int64}; img_id = 文件名末尾数字。

用法:
  python tools/data/build_image_latents.py --csv 5script/train_fame3_sym_full.csv \
      --out final_latents_fame_sym --batch 32 --shard-size 5000
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SCALING_FACTOR = 0.18215


def read_img(path):
    img = Image.open(path).convert("L")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    a = np.stack([a, a, a], axis=-1)
    return np.transpose(a, (2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    tasks = []
    for r in rows:
        p = r["image_path"]
        m = re.search(r"(\d+)\.png$", p)
        if not m:
            print(f"  [skip] bad path: {p}")
            continue
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.isfile(full):
            print(f"  [skip] missing: {full}")
            continue
        tasks.append((full, int(m.group(1))))
    print(f"[csv] {len(rows)} rows, valid {len(tasks)}", flush=True)

    from diffusers import AutoencoderKL
    dev = args.device
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    shard_lat, shard_id, n_shard, done = [], [], 0, 0

    def flush():
        nonlocal shard_lat, shard_id, n_shard
        if not shard_lat:
            return
        np.savez_compressed(os.path.join(args.out, f"shard_{n_shard:05d}.npz"),
                            latents=np.stack(shard_lat).astype(np.float16),
                            img_ids=np.array(shard_id, dtype=np.int64))
        n_shard += 1
        shard_lat, shard_id = [], []

    with torch.no_grad():
        for i in range(0, len(tasks), args.batch):
            b = tasks[i:i + args.batch]
            xs = torch.from_numpy(np.stack([read_img(p) for p, _ in b], 0)).to(dev)
            lat = vae.encode(xs).latent_dist.sample() * SCALING_FACTOR
            if lat.shape[-1] != 32:
                lat = torch.nn.functional.interpolate(lat, size=32, mode="bilinear")
            lat = lat.float().cpu().numpy().astype(np.float16)
            for k, (_, iid) in enumerate(b):
                shard_lat.append(lat[k])
                shard_id.append(iid)
                done += 1
            if len(shard_lat) >= args.shard_size:
                flush()
            if (i // args.batch) % 128 == 0 or i + args.batch >= len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {done/max(el,1e-9):.0f}/s  "
                      f"gpu={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
    flush()
    print(f"[done] {done} latents, {n_shard} shards -> {args.out}/ in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
