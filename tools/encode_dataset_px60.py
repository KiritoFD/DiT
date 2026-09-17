# -*- coding: utf-8 -*-
"""encode_dataset_px60.py — 为 fame-kxl-tj-px60 编码 image / 标准字 latent.

输入: assets/train_fame-kxl-tj-px60.csv  (列含 image_path / std_path, id 相同)
产物:
  data/fame-kxl-tj-px60/shards_img/   image latent shards (img_id-keyed, float16)
  data/fame-kxl-tj-px60/shards_std/   标准字 latent shards (img_id-keyed, float16)

encode_csv 只读 csv 的 image_path 列, 故分别写临时 csv 传入。
用法: python tools/encode_dataset_px60.py [--which img|std|both]
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.vae_io import encode_csv  # noqa: E402

NAME = "fame-kxl-tj-px60"
CSV = f"assets/train_{NAME}.csv"
BASE = f"data/{NAME}"
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
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"[encode:{NAME}] {CSV}: {len(rows)} 行")
    mi = sum(1 for r in rows if not os.path.exists(r["image_path"]))
    ms = sum(1 for r in rows if not os.path.exists(r["std_path"]))
    print(f"  缺图 {mi}, 缺标准字 {ms}")
    assert mi == 0 and ms == 0, "有缺失文件, 先修数据"

    jobs = []
    if a.which in ("img", "both"):
        jobs.append(("img", "image_path", f"{BASE}/shards_img", "/tmp/_px60_img.csv"))
    if a.which in ("std", "both"):
        jobs.append(("std", "std_path", f"{BASE}/shards_std", "/tmp/_px60_std.csv"))

    for name, key, out, tmp in jobs:
        dump_tmp(rows, key, tmp)
        have = len(glob.glob(os.path.join(out, "shard_*.npz")))
        expect = (len(rows) + 4999) // 5000
        if have >= expect:
            print(f"[{name}] skip: {out} 已有 {have} shards")
            continue
        print(f"[{name}] encoding {len(rows)} -> {out} ...", flush=True)
        encode_csv(tmp, out, transform="gray", vae_path=VAE, batch=128,
                   shard_size=5000, workers=12, device="cuda")
        tot = 0
        for sp in glob.glob(os.path.join(out, "shard_*.npz")):
            with np.load(sp) as d:
                tot += d["img_ids"].shape[0]
        print(f"[{name}] DONE {out}: {tot} latents", flush=True)


if __name__ == "__main__":
    main()
