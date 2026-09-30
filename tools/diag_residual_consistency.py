# -*- coding: utf-8 -*-
"""diag_residual_consistency.py — 残差 r=x0-g 是"真实逐字结构"还是"latent 噪声"?

上一测发现: 风格槽位只能解释残差的 0.62%, 99.38% 是同槽位内逐字差异。
但有两种截然不同的解释:
  (a) 真实: 残差是 (书家,字) 的确定签名 -> 原则上**可学**(字符身份由 g 携带),
            学不会就是容量/优化问题 -> 该加容量
  (b) 噪声: 残差是 VAE latent 的配准/走样噪声 -> **不可学**, 谁都学不会 -> 该改目标

判据: 同一 (槽位,字) 有多张样本时, 看这些样本的残差彼此有多一致:
  · 组内一致 (within-combo cosine) 显著 > 组间 (between-combo cosine) -> (a) 真实
  · 组内 ≈ 组间 ≈ 0                                                  -> (b) 噪声
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_map(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            lat = z["latents"]
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(lat[j], dtype=np.float64).ravel()
    return mp


def unit(v):
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_top10_style23.csv")
    ap.add_argument("--tgt", default="data/top10_style23/shards_gtskel_w3")
    ap.add_argument("--cond", default="data/top10_style23/shards_std")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    tgt = load_map(a.tgt)
    cond = load_map(a.cond)
    print(f"csv {len(rows)} | tgt {len(tgt)} | cond {len(cond)}")

    by_key = defaultdict(list)
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if not m:
            continue
        i = int(m.group(1))
        if i not in tgt or i not in cond:
            continue
        key = (r.get("slot_name", "?"), r.get("character", "?"))
        by_key[key].append(unit(tgt[i] - cond[i]))

    multi = {k: v for k, v in by_key.items() if len(v) >= 2}
    print(f"(槽位,字) 组合 {len(by_key)}, 其中样本数>=2 的 {len(multi)}")

    within, between = [], []
    keys = list(by_key)
    rng = np.random.default_rng(0)
    for k, vs in multi.items():
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                within.append(float(vs[i] @ vs[j]))
        # 随机找一个不同 key 做组间对照
        for _ in range(min(3, len(vs))):
            k2 = keys[rng.integers(len(keys))]
            if k2 == k:
                continue
            v2 = by_key[k2][rng.integers(len(by_key[k2]))]
            between.append(float(vs[0] @ v2))

    w, b = np.array(within), np.array(between)
    print(f"\n残差方向余弦 (残差已单位化):")
    print(f"  组内 (同一 槽位+字, 不同样本) n={len(w)}  mean={w.mean():+.4f}  "
          f"std={w.std():.4f}")
    print(f"  组间 (不同 槽位/字)          n={len(b)}  mean={b.mean():+.4f}  "
          f"std={b.std():.4f}")
    gap = w.mean() - b.mean()
    print(f"\n  ★ 组内−组间 = {gap:+.4f}")
    if gap > 0.15:
        print("  -> (a) 残差是**真实的 (书家,字) 签名**: 一致性强。学不会 = 容量/优化问题 -> 该加容量/注入层数。")
    elif gap < 0.05:
        print("  -> (b) 残差基本是**噪声**: 组内并不比组间更一致。任何模型都学不会 -> 该改目标(别预测 GT 骨架)。")
    else:
        print("  -> 介于两者之间: 有部分真实结构但被大量噪声淹没。")


if __name__ == "__main__":
    main()
