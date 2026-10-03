#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量 predskel latent 与 GT 骨架 latent 的幅度 (定位"发灰"是不是幅度不足)"""
import glob
import os
import re
import sys

import numpy as np

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")


def load(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(z["latents"][j], dtype=np.float32)
    return mp


ids = []
import csv
for r in csv.DictReader(open("assets/eval_top10_seen_20.csv", encoding="utf-8")):
    m = re.search(r"(\d+)\.png", r["image_path"])
    if m:
        ids.append(int(m.group(1)))
    if len(ids) >= 20:
        break

DIRS = [("我们的 predskel", "data/top10_style23/predskel_fmdit_seen20_p32"),
        ("GT 骨架 (v26 用)", "data/top10_style23/shards_aux_skel3"),
        ("std 骨架 (v25/基线用)", "data/top10_style23/shards_std")]
ref = None
for nm, d in DIRS:
    mp = load(d)
    have = [i for i in ids if i in mp]
    if not have:
        print(f"{nm}: 无匹配 id ({d})")
        continue
    A = np.stack([mp[i] for i in have])              # (n,4,32,32)
    flat = A.reshape(len(have), -1)
    nrm = np.linalg.norm(flat, axis=1)
    print(f"{nm:>22}: n={len(have)}  |lat| mean={nrm.mean():.3f} "
          f"std={nrm.std():.3f}  per-dim std={flat.std():.4f}  "
          f"mean={flat.mean():+.4f}")
    if ref is None:
        ref = nrm.mean()
    else:
        print(f"{'':>22}  -> 幅度比 vs 第一个 = {nrm.mean()/ref:.3f}")
