# -*- coding: utf-8 -*-
"""局部替换：只 encode clean 修复图(1430张)，替换进旧 shards，写 data/latents/final_latents_fame_clean。

用户要求：不全部重新 encode，npz 可编辑，encode 需修复部分然后替换。

流程
----
1. 读 train_fame_clean.csv，找指向 data/imgs/final_imgs_256_clean 的 1430 行（img_id + 路径）。
2. GPU VAE encode 这 1430 张 clean 图 → (4,32,32) latent。
3. 读旧 shards (data/latents/final_latents_fame/shard_*.npz)。
4. 按 img_id 找到每张 clean 图在旧 shard 里的位置，替换 latent。
5. 写新 shards 到 data/latents/final_latents_fame_clean/（格式不变：latents fp16, img_ids int64）。
6. 校验：所有 clean id 已替换，其余 id 保留原 latent，id 集合一致。
"""
import os, sys, csv, re, time, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
DEVICE = torch.device("cuda")
CSV = "assets/train_fame_clean.csv"
OLD_DIR = "data/latents/final_latents_fame"
OUT_DIR = "data/latents/final_latents_fame_clean"
VAE_PATH = "data/pretrained/sd-vae-ft-ema"
SCALING_FACTOR = 0.18215
VAE_BS = 32


def read_img(path):
    img = Image.open(path).convert("L").resize((256, 256), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    a = np.stack([a, a, a], axis=-1)
    return np.transpose(a, (2, 0, 1))


def main():
    # 1. 收集 clean 修复图
    tasks = []
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if "clean" not in r["image_path"]:
            continue
        m = re.search(r"(\d+)\.png$", r["image_path"])
        tasks.append((int(m.group(1)), os.path.join(ROOT, r["image_path"])))
    print(f"clean images to re-encode: {len(tasks)}")

    # 2. GPU encode
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    new_lat = {}   # img_id -> (4,32,32) fp16
    t0 = time.time()
    for i in range(0, len(tasks), VAE_BS):
        batch = tasks[i:i+VAE_BS]
        imgs = []
        ids = []
        for iid, p in batch:
            if not os.path.isfile(p):
                print(f"  [missing] {p}")
                continue
            imgs.append(read_img(p))
            ids.append(iid)
        if not imgs:
            continue
        xs = torch.from_numpy(np.stack(imgs, 0)).to(DEVICE)
        with torch.no_grad():
            lat = vae.encode(xs).latent_dist.sample()
            lat = lat * SCALING_FACTOR
            if lat.shape[-1] != 32:
                lat = torch.nn.functional.interpolate(lat, size=32, mode="bilinear")
            lat = lat.float().cpu().numpy().astype(np.float16)
        for k, iid in enumerate(ids):
            new_lat[iid] = lat[k]
        print(f"  encoded {min(i+VAE_BS,len(tasks))}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"encoded {len(new_lat)} latents ({time.time()-t0:.0f}s)")

    # 3. 读旧 shards，替换，写新 shards
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_files = sorted(glob.glob(os.path.join(OLD_DIR, "shard_*.npz")))
    print(f"old shards: {len(shard_files)}")
    n_replaced = 0
    for sp in shard_files:
        d = np.load(sp)
        lat = d["latents"]          # (N,4,32,32) fp16
        ids = d["img_ids"]          # (N,) int64
        lat_new = lat.copy()
        replaced_here = 0
        for idx, iid in enumerate(ids.tolist()):
            if iid in new_lat:
                lat_new[idx] = new_lat[iid]
                replaced_here += 1
        if replaced_here:
            n_replaced += replaced_here
        d.close()
        out = os.path.join(OUT_DIR, os.path.basename(sp))
        np.savez_compressed(out, latents=lat_new, img_ids=ids)
        print(f"  {os.path.basename(sp)}: replaced {replaced_here}/{len(ids)} -> {out}")
    print(f"total replaced: {n_replaced}/{len(new_lat)}")

    # 4. 校验
    new_ids = set()
    for sp in sorted(glob.glob(os.path.join(OUT_DIR, "shard_*.npz"))):
        d = np.load(sp)
        new_ids.update(int(x) for x in d["img_ids"])
        d.close()
    old_ids = set()
    for sp in shard_files:
        d = np.load(sp)
        old_ids.update(int(x) for x in d["img_ids"])
        d.close()
    print(f"old ids={len(old_ids)} new ids={len(new_ids)} match={old_ids==new_ids}")
    print(f"all clean ids present: {set(new_lat.keys()) <= new_ids}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
