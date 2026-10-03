#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w3 目标 == 主干的条件? 比 shards_gtskel_w3 与 shards_aux_skel3 的 latent。"""
import glob
import os

import numpy as np

os.chdir("/root/Workspace/xy/DiT")


def load(d, cap=3000):
    mp = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(z["latents"][j], np.float32)
        if len(mp) >= cap:
            break
    return mp


A = load("data/top10_style23/shards_gtskel_w3")
B = load("data/top10_style23/shards_aux_skel3")
C = load("data/top10_style23/shards_std")
print(f"gtskel_w3 {len(A)} | aux_skel3 {len(B)} | shards_std {len(C)}")


def cmp(n1, m1, n2, m2):
    cc = sorted(set(m1) & set(m2))
    if not cc:
        print(f"{n1} vs {n2}: 无共同 id")
        return
    a = np.stack([m1[i] for i in cc[:200]])
    b = np.stack([m2[i] for i in cc[:200]])
    d = np.abs(a - b).mean()
    cos = np.mean([float(x.ravel() @ y.ravel() /
                         (np.linalg.norm(x.ravel()) * np.linalg.norm(y.ravel()) + 1e-9))
                   for x, y in zip(a, b)])
    print(f"{n1:12} vs {n2:12}: 共同 {len(cc)}  逐元素cos {cos:+.5f}  |Δ| {d:.6f}  "
          f"{'★ 同一份' if cos > 0.999 and d < 1e-6 else '不同'}")


cmp("gtskel_w3", A, "aux_skel3", B)
cmp("gtskel_w3", A, "shards_std", C)
cmp("aux_skel3", B, "shards_std", C)
