# -*- coding: utf-8 -*-
"""把 50k 的 aux 图（canny / skel3）VAE 编码成 latent shard。

## transform 选 gray（不反转）
`tools/gen_aux_50k.py` 产出的 PNG 已经是**白底(255)黑线(0)**（doc58 的新规范）。
`_tf_gray` 直接读 -> 白底=+1、线=-1 ✓
**不能**用 `_tf_canny`：它会**重新算一遍 Canny** 并强制反转 —— 输入已是边缘图，
二次 Canny 只会出垃圾。

## 输出（与旧 aux 目录同名风格，但按 50k 的 id）
    data/50k/shards_aux_canny   (对应旧 data/aux/aux_canny_latents_base)
    data/50k/shards_aux_skel3   (对应旧 data/aux/inst_skel_latents_px60)

用法:
    python tools/encode_aux_50k.py --which both
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
from src.data.vae_io import encode_csv  # noqa: E402

CSV = "assets/train_50k_v2.csv"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
JOBS = {
    "canny": ("data/50k/aux_canny", "data/50k/shards_aux_canny"),
    "skel3": ("data/50k/aux_skel3", "data/50k/shards_aux_skel3"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both",
                    choices=["canny", "skel3", "both"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    names = ["canny", "skel3"] if a.which == "both" else [a.which]
    for name in names:
        src_dir, out_dir = JOBS[name]
        tmp = f"/tmp/_aux50k_{name}.csv"
        n_missing = 0
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "character", "script", "calligrapher"])
            for r in rows:
                iid = os.path.basename(r["image_path"]).rsplit(".", 1)[0]
                p = f"{src_dir}/{iid}.png"
                if not os.path.exists(p):
                    n_missing += 1
                    continue
                w.writerow([p, r.get("character", ""), r.get("script", ""),
                            r.get("calligrapher", "")])
        have = len(glob.glob(os.path.join(out_dir, "shard_*.npz")))
        print(f"[aux-enc] {name}: {len(rows)} 行, 缺图 {n_missing}, "
              f"已有 {have} shards", flush=True)
        if have >= (len(rows) + 4999) // 5000:
            print(f"[aux-enc] {name} 已完成，跳过", flush=True)
            continue
        encode_csv(tmp, out_dir, transform="gray", vae_path=VAE,
                   batch=a.batch, shard_size=5000, workers=a.workers,
                   device="cuda")
        tot = sum(np.load(sp)["img_ids"].shape[0]
                  for sp in glob.glob(os.path.join(out_dir, "shard_*.npz")))
        print(f"[aux-enc] {name} DONE {out_dir}: {tot} latents", flush=True)


if __name__ == "__main__":
    main()
