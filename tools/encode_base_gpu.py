# -*- coding: utf-8 -*-
"""
encode_base_gpu.py — fame-tj-uc base GPU encode (训练已停, 独占 GPU).

  img        — base csv GT 图 (transform gray)          -> data/latents/final_latents_base_shards
  aux_skel3  — 编 data/skel/final_skel3_base/{id}.png   -> data/skel/aux_skel3_latents_base
  aux_canny  — 编 data/aux/final_canny_base/{id}.png    -> data/aux/aux_canny_latents_base
  std_expand — data/skel/std_skel3_latents_base (uid 8600000+, 已完成) 按 base csv 行
               展开为 img_id-keyed shards (写同目录 shard_{n:05d}.npz)

实测: GPU encode bs128 ≈ 160 img/s → 165k 张 ≈ 18 分钟.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

from src.data.vae_io import encode_csv

VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"


def make_png_csv(png_dir):
    """把 PNG 目录变成 vae_io 可读的 csv (image_path)。"""
    rows = []
    for p in sorted(glob.glob(os.path.join(png_dir, "*.png"))):
        rows.append({"image_path": p})
    tmp = os.path.join("/tmp", os.path.basename(png_dir) + ".csv")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return tmp, len(rows)


def shard_count(d):
    return len(glob.glob(os.path.join(d, "shard_*.npz")))


def std_expand():
    """std_skel latent (uid-keyed) -> 按 base csv 行展开 img_id-keyed shards."""
    key2uid = {}
    for r in csv.DictReader(open("data/skel/std_skel3_base_key2uid.csv", encoding="utf-8")):
        key2uid[(r["script"], r["character"])] = int(r["uid"])
    uid2lat = {}
    for sp in glob.glob("data/skel/std_skel3_latents_base/shard_*.npz"):
        with np.load(sp) as d:
            for j, uid in enumerate(d["img_ids"]):
                uid2lat[int(uid)] = d["latents"][j]
    print(f"[std_expand] uid latents: {len(uid2lat)}", flush=True)
    out_dir = "data/skel/std_skel3_latents_base"
    os.makedirs(out_dir, exist_ok=True)
    buf, ids, n_shard = [], [], 0
    n_have = 0
    for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        u = key2uid.get((r["script"], r["character"]))
        if u is None or u not in uid2lat:
            continue
        buf.append(uid2lat[u])
        ids.append(iid)
        n_have += 1
        if len(buf) >= 5000:
            np.savez(os.path.join(out_dir, f"expand_{n_shard:05d}.npz"),
                     latents=np.stack(buf).astype(np.float16),
                     img_ids=np.array(ids, dtype=np.int64))
            n_shard += 1
            buf, ids = [], []
    if buf:
        np.savez(os.path.join(out_dir, f"expand_{n_shard:05d}.npz"),
                 latents=np.stack(buf).astype(np.float16),
                 img_ids=np.array(ids, dtype=np.int64))
    print(f"[std_expand] {n_have} rows -> {n_shard+1} shards", flush=True)


def main():
    # 1) img latents (已完成: 11 shards)
    if shard_count("data/latents/final_latents_base_shards") >= 11:
        print("=== img: skip (done) ===", flush=True)
    else:
        print("=== img ===", flush=True)
        encode_csv("assets/train_base_noaug.csv", "data/latents/final_latents_base_shards",
                   transform="gray", vae_path=VAE_PATH, batch=128, shard_size=5000,
                   workers=12, device="cuda")

    # 2) aux skel3 (从已落盘 PNG 编码, transform gray: 黑线白底线图直通)
    print("=== aux_skel3 ===", flush=True)
    tmp, n = make_png_csv("data/skel/final_skel3_base")
    encode_csv(tmp, "data/skel/aux_skel3_latents_base", transform="gray",
               vae_path=VAE_PATH, batch=128, shard_size=5000, workers=12, device="cuda")

    # 3) aux canny
    print("=== aux_canny ===", flush=True)
    tmp, n = make_png_csv("data/aux/final_canny_base")
    encode_csv(tmp, "data/aux/aux_canny_latents_base", transform="gray",
               vae_path=VAE_PATH, batch=128, shard_size=5000, workers=12, device="cuda")

    # 4) std skel 展开
    print("=== std_expand ===", flush=True)
    std_expand()


if __name__ == "__main__":
    main()
