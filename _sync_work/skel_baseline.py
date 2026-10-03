#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算 stage1 的骨架级基线: std 骨架 vs GT 骨架 的 ssim (与 in-mem-eval 同尺)。

stage1 的 in-mem-eval 报的是"生成骨架 vs GT骨架"的 ssim (strict 0.6034)。
没有基线就无法判断它到底做得好不好 —— 这个脚本给"什么都不做"的同一个数。
"""
import csv
import glob
import os
import re
import sys

import numpy as np
import torch as th

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
from skimage.metrics import structural_similarity as ssim  # noqa: E402


def load_idx(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


def get(idx, i):
    f, j = idx[i]
    with np.load(f) as z:
        return np.asarray(z["latents"][j], np.float32)


ids = []
for r in csv.DictReader(open("assets/eval_top10_strict_subset84.csv", encoding="utf-8")):
    m = re.search(r"(\d+)\.png$", r.get("image_path", ""))
    if not m:
        continue
    i = int(m.group(1))
    if os.path.exists(f"data/top10_style23/gt_skel_png/{i:06d}.png"):
        ids.append(i)
ids = ids[:84]
print(f"strict84 条数: {len(ids)}")

std_i = load_idx("data/top10_style23/shards_std")
gt_i = load_idx("data/top10_style23/shards_gtskel_w3")
both = [i for i in ids if i in std_i and i in gt_i]
print(f"两边都有: {len(both)}")

from diffusers.models import AutoencoderKL
dev = "cuda"
vae = AutoencoderKL.from_pretrained(
    "data/pretrained/pretrained_models/sd-vae-ft-ema").to(dev).eval()


def dec(lats):
    z = th.from_numpy(np.stack(lats)).to(dev)
    out = []
    with th.no_grad(), th.autocast("cuda", dtype=th.float16):
        for s in range(0, z.shape[0], 16):
            out.append(vae.decode((z[s:s + 16] / 0.18215).half()).sample.mean(1)
                       .float().cpu().numpy())
    return np.concatenate(out)


ds = dec([get(std_i, i) for i in both])
dg = dec([get(gt_i, i) for i in both])


def norm(x):
    x = x - x.min()
    return x / max(x.max(), 1e-9)


S = [ssim(norm(ds[k]), norm(dg[k]), data_range=1.0) for k in range(len(both))]
ink_s = [(ds[k] < 0).mean() / max((dg[k] < 0).mean(), 1e-9) for k in range(len(both))]
print(f"\n【基线】std骨架 vs GT骨架  ssim = {np.mean(S):.4f}  (n={len(S)})")
print(f"         墨量比 std/GT = {np.mean(ink_s):.3f}")
print(f"\n对照: stage1 报的 strict 骨架级 ssim = 0.6034")
print(f"      差值 = {0.6034 - np.mean(S):+.4f}")
