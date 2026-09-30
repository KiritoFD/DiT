#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ink_report.py — 报若干 shard 目录的**解码墨量** (对 GT 骨架归一)。

诊断漂白用: 墨量比 <0.7 即明显漂白 (骨架该是二值的, 淡影说明 latent 落在均值处)。

用法:
  python tools/ink_report.py --csv assets/eval_top10_seen_20.csv \
      --ref data/top10_style23/shards_aux_skel3 \
      --dirs "w7b1.0:data/.../predskel_dit_seen20_w7b1.0,w7b1.5:..."
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def load_map(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(z["latents"][j], dtype=np.float32)
    return mp


def csv_ids(p, n):
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            out.append(int(m.group(1)))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dirs", required=True, help="tag:dir,tag:dir,...")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    a = ap.parse_args()
    dev = "cuda"

    ids = csv_ids(a.csv, a.n)
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()

    def ink_of(mp, want):
        have = [i for i in ids if i in mp]
        if not have:
            return None, 0
        t = th.from_numpy(np.stack([mp[i] for i in have])).to(dev)
        with th.no_grad():
            b = (vae.decode(t / 0.18215).sample.mean(1) < 0).cpu().numpy()
        return float(b.mean()), len(have)

    ref = load_map(a.ref)
    ri, rn = ink_of(ref, ids)
    print(f"GT 参考 {a.ref}: ink={ri:.4f} (n={rn})")
    for spec in a.dirs.split(","):
        if not spec.strip():
            continue
        tag, d = spec.split(":")
        if not os.path.isdir(d):
            print(f"  {tag:>12}: 目录缺失")
            continue
        v, n = ink_of(load_map(d), ids)
        if v is None:
            print(f"  {tag:>12}: 无匹配 id")
        else:
            print(f"  {tag:>12}: ink={v:.4f}  ratio={v/max(ri,1e-9):.2f}  (n={n})")


if __name__ == "__main__":
    main()
