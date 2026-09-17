# -*- coding: utf-8 -*-
"""encode_clean_v1.py — 对 clean_v1 的"图"和"标准字"分别 encode, id 键对齐.

输入: assets/train_clean_v1_final.csv   列含 image_path / std_path
产物:
  data/clean_v1/shards_img/  image latent shards  (img_id-keyed, float16)
  data/clean_v1/shards_std/  标准字 latent shards (img_id-keyed, float16)
  —— 两者 img_id **相同**(都取自 <id>.png), 训练时按同一 id 取 g 条件。

encode_csv 只认 csv 的 image_path 列, 故这里各写一份临时 csv 传入。
用法: python tools/encode_clean_v1.py [--which img|std|both]
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

CSV = "assets/train_clean_v1_final.csv"
BASE = "data/clean_v1"
OUT_IMG = f"{BASE}/shards_img"
OUT_STD = f"{BASE}/shards_std"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"


def dump_tmp(rows, key, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "character", "script", "calligrapher"])
        for r in rows:
            w.writerow([r[key], r.get("character", ""), r.get("script", ""),
                        r.get("calligrapher", "")])
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["img", "std", "both"])
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"[encode] {CSV}: {len(rows)} 行")
    miss_i = sum(1 for r in rows if not os.path.exists(r["image_path"]))
    miss_s = sum(1 for r in rows if not os.path.exists(r["std_path"]))
    print(f"  缺图 {miss_i}, 缺标准字 {miss_s}")
    assert miss_i == 0, "有缺失的图, 请先修"
    assert miss_s == 0, "有缺失的标准字, 请先修"

    jobs = []
    if a.which in ("img", "both"):
        jobs.append(("img", "image_path", OUT_IMG, "/tmp/_clean_img.csv"))
    if a.which in ("std", "both"):
        jobs.append(("std", "std_path", OUT_STD, "/tmp/_clean_std.csv"))

    for name, key, out, tmp in jobs:
        n = dump_tmp(rows, key, tmp)
        have = len(glob.glob(os.path.join(out, "shard_*.npz")))
        expect = (n + 4999) // 5000
        if have >= expect:
            print(f"[{name}] skip: {out} 已有 {have} shards (>= {expect})")
            continue
        print(f"[{name}] encoding {n} -> {out} ...", flush=True)
        encode_csv(tmp, out, transform="gray", vae_path=VAE, batch=128,
                   shard_size=5000, workers=12, device="cuda")
        tot = 0
        for sp in glob.glob(os.path.join(out, "shard_*.npz")):
            with np.load(sp) as d:
                tot += d["img_ids"].shape[0]
        print(f"[{name}] DONE {out}: {tot} latents", flush=True)


if __name__ == "__main__":
    main()
