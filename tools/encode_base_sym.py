# -*- coding: utf-8 -*-
"""encode_base_sym.py — 为 158K 全对称增强数据 (assets/train_base_sym.csv) 编码 image latent.

产物: data/latents/final_latents_base_sym  (img_id-keyed shards, float16)
     —— 全量 158,814 行 (base 原图 54,892 + base_sym 增强 103,922) 一次编完,
        训练时一个目录覆盖全量, 不用再合并。

实测: GPU bs128 ≈ 160 img/s -> 158k 张 ≈ 17 分钟。
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

from src.data.vae_io import encode_csv  # noqa: E402

VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"
CSV = "assets/train_base_sym.csv"
OUT = "data/latents/final_latents_base_sym"


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"[csv] {CSV} rows={len(rows)}", flush=True)
    have = len(glob.glob(os.path.join(OUT, "shard_*.npz")))
    if have >= 32:
        print(f"[skip] {OUT} 已有 {have} shards", flush=True)
        return
    encode_csv(CSV, OUT, transform="gray", vae_path=VAE_PATH, batch=128,
               shard_size=5000, workers=12, device="cuda")
    n = len(glob.glob(os.path.join(OUT, "shard_*.npz")))
    tot = 0
    for sp in glob.glob(os.path.join(OUT, "shard_*.npz")):
        with np.load(sp) as d:
            tot += d["img_ids"].shape[0]
    print(f"[done] {OUT}: {n} shards, {tot} latents", flush=True)


if __name__ == "__main__":
    main()
