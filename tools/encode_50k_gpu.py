# -*- coding: utf-8 -*-
"""50k 数据集的 **GPU** VAE 编码（img + std 两套 latent shard）。

## 为什么用 GPU 版
之前 GPU 被训练占用时写了 CPU 版（tools/cpu_encode_50k.py）。现在 GPU 空了，
GPU 编码比 CPU 快几十倍 —— 直接复用 `src.data.vae_io.encode_csv`，语义与旧数据集
（encode_dataset_px60.py）**完全一致**：transform="gray"、fp16 latent、shard_*.npz。

## ⚠ 与旧数据集的关键差异：img_id
新数据集文件名是 **6 位补零**（`data/50k/imgs/000000.png`），
`extract_img_id` 会解析出 `0, 1, 2, ...`；
而旧数据集是 `77.png -> 77`。

**这会让 REPA 的 DINO 缓存静默错位**（缓存按旧 id 建）：
新 id 77 会命中旧 id 77 的特征，但那是**另一张图**。
所以 v13 **必须重建 DINO 缓存**（见 tools/build_dino_cache.py），不能复用 base_sym_v1。
本脚本只负责 VAE latent，DINO 缓存是独立一步。

## 用法
    python tools/encode_50k_gpu.py --which both
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.data.vae_io import encode_csv  # noqa: E402

CSV = "assets/train_50k.csv"
BASE = "data/50k"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"


def dump_tmp(rows, key, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "character", "script", "calligrapher"])
        for r in rows:
            w.writerow([r[key], r.get("character", ""), r.get("script", ""),
                        r.get("calligrapher", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["img", "std", "both"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"[encode:50k] {CSV}: {len(rows)} 行", flush=True)
    mi = sum(1 for r in rows if not os.path.exists(r["image_path"]))
    ms = sum(1 for r in rows if not os.path.exists(r["std_path"]))
    print(f"  缺图 {mi}, 缺标准字 {ms}", flush=True)
    assert mi == 0 and ms == 0, "有缺失文件, 先修数据"

    jobs = []
    if a.which in ("img", "both"):
        jobs.append(("img", "image_path", f"{BASE}/shards_img", "/tmp/_50k_img.csv"))
    if a.which in ("std", "both"):
        jobs.append(("std", "std_path", f"{BASE}/shards_std", "/tmp/_50k_std.csv"))

    for name, key, out, tmp in jobs:
        dump_tmp(rows, key, tmp)
        have = len(glob.glob(os.path.join(out, "shard_*.npz")))
        expect = (len(rows) + 4999) // 5000
        if have >= expect:
            print(f"[{name}] skip: {out} 已有 {have} shards", flush=True)
            continue
        print(f"[{name}] encoding {len(rows)} -> {out} ...", flush=True)
        encode_csv(tmp, out, transform="gray", vae_path=VAE, batch=a.batch,
                   shard_size=5000, workers=a.workers, device="cuda")
        tot = 0
        for sp in glob.glob(os.path.join(out, "shard_*.npz")):
            with np.load(sp) as d:
                tot += d["img_ids"].shape[0]
        print(f"[{name}] DONE {out}: {tot} latents", flush=True)


if __name__ == "__main__":
    main()
