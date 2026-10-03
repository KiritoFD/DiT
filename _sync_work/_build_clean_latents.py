# -*- coding: utf-8 -*-
"""从 clean CSV 构建 VAE latent shards（训练集）。

读 ``assets/train_fame_clean.csv``（image_path 可指向 data/imgs/final_imgs_256 或
data/imgs/final_imgs_256_clean），对每张图 VAE encode 成 (4,32,32) latent，按 shard 保存
到 ``data/latents/final_latents_fame_clean/``。格式与 ``data/latents/final_latents_fame/`` 一致：
  shard_XXXXX.npz { latents: (N,4,32,32) fp16, img_ids: (N,) int64 }

img_id = image_path 文件名数字（clean 修复图沿用原图 id）。
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
OUT_DIR = "data/latents/final_latents_fame_clean"
VAE_PATH = "data/pretrained/sd-vae-ft-ema"
SCALING_FACTOR = 0.18215
SHARD_SIZE = 5000
VAE_BS = 32


def load_model():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def read_img(path):
    """读图 → (3,256,256) float32 [-1,1]。"""
    img = Image.open(path).convert("L").resize((256, 256), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    a = np.stack([a, a, a], axis=-1)
    return np.transpose(a, (2, 0, 1))


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"rows={len(rows)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 预处理 img_path -> abs + img_id
    tasks = []
    for r in rows:
        p = r["image_path"]
        m = re.search(r"(\d+)\.png$", p)
        if not m:
            print(f"  [skip] bad path: {p}")
            continue
        iid = int(m.group(1))
        tasks.append((os.path.join(ROOT, p), iid))
    print(f"valid tasks={len(tasks)}")

    vae = load_model()
    t0 = time.time()
    lat_list = []
    id_list = []
    for i in range(0, len(tasks), VAE_BS):
        batch = tasks[i:i+VAE_BS]
        imgs = []
        ids = []
        for p, iid in batch:
            if not os.path.isfile(p):
                print(f"  [skip] missing {p}")
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
            lat_list.append(lat[k])
            id_list.append(iid)
        if (i // VAE_BS) % 256 == 0 or i + VAE_BS >= len(tasks):
            el = time.time() - t0
            print(f"  {i+len(imgs)}/{len(tasks)}  {i/el:.0f}/s  gpu={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
    print(f"encoded {len(lat_list)} latents in {time.time()-t0:.0f}s")

    # 写 shards
    lat_arr = np.stack(lat_list, 0)          # (N,4,32,32) fp16
    id_arr = np.array(id_list, dtype=np.int64)
    n_shards = (len(lat_arr) + SHARD_SIZE - 1) // SHARD_SIZE
    for s in range(n_shards):
        sl = s * SHARD_SIZE
        el_ = min(sl + SHARD_SIZE, len(lat_arr))
        out = os.path.join(OUT_DIR, f"shard_{s:05d}.npz")
        np.savez_compressed(out, latents=lat_arr[sl:el_], img_ids=id_arr[sl:el_])
        print(f"  [{s+1}/{n_shards}] {out}: {lat_arr[sl:el_].shape}")

    # 校验
    shard_ids = set()
    for sp in sorted(glob.glob(os.path.join(OUT_DIR, "shard_*.npz"))):
        d = np.load(sp)
        shard_ids.update(int(x) for x in d["img_ids"])
        d.close()
    csv_ids = set(iid for _, iid in tasks)
    missing = csv_ids - shard_ids
    dup = len(shard_ids) != len(csv_ids)
    print(f"csv_ids={len(csv_ids)} shard_ids={len(shard_ids)} missing={len(missing)} dup={dup}")
    if not missing and not dup:
        print("ALL OK ✓")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
