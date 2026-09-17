# -*- coding: utf-8 -*-
"""encode_std_px60.py — 把重建后的标准字骨架 PNG 编码成 g 条件 latent shards.

产出两个目录 (都必须按各自的 img_id 建, 训练/评测分开):
  data/fame-kxl-tj-px60/shards_std       <- 训练 g, id = train csv 的 img_id
  data/fame-kxl-tj-px60/shards_std_eval  <- 评测 g, id = eval csv 的 img_id
     (必须配 config 的 eval_skel_latent_shards_dir; 留空会退回训练目录 ->
      strict 命中 0 -> g 全零 -> decode(0) 出灰黄棕、指标失真)

用法: python tools/encode_std_px60.py
"""
import csv
import glob
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.vae_io import encode_csv  # noqa: E402

NAME = "fame-kxl-tj-px60"
BASE = f"data/{NAME}"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
JOBS = [
    (f"assets/train_{NAME}.csv", "std_path", f"{BASE}/shards_std"),
    (None, "std_eval", f"{BASE}/shards_std_eval"),
]


def shard_total(d):
    t = 0
    for sp in glob.glob(os.path.join(d, "shard_*.npz")):
        with np.load(sp) as z:
            t += z["img_ids"].shape[0]
    return t


def main():
    train = list(csv.DictReader(open(f"assets/train_{NAME}.csv", encoding="utf-8")))

    for csv_src, key, out in JOBS:
        os.makedirs("/tmp", exist_ok=True)
        tmp = f"/tmp/_enc_{key}.csv"
        if csv_src:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path", "character", "script", "calligrapher"])
                n = 0
                for r in train:
                    p = r[key]
                    if os.path.exists(p):
                        w.writerow([p, r.get("character", ""), r.get("script", ""),
                                    r.get("calligrapher", "")])
                        n += 1
        else:
            pngs = sorted(glob.glob(f"{BASE}/std_eval/*.png"))
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path", "character", "script", "calligrapher"])
                for p in pngs:
                    w.writerow([p, "", "", ""])
            n = len(pngs)
        if not n:
            print(f"[{key}] 无输入, 跳过")
            continue
        if os.path.isdir(out):
            shutil.rmtree(out)
        print(f"[{key}] encoding {n} -> {out} ...", flush=True)
        encode_csv(tmp, out, transform="gray", vae_path=VAE, batch=64,
                   shard_size=5000, workers=8, device="cuda")
        print(f"[{key}] DONE {out}: {shard_total(out)} latents", flush=True)


if __name__ == "__main__":
    main()
