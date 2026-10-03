#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对两套「标准骨架 latent」是否同一命名空间。
若 top10_style23/shards_std 与 50k_v2_glyph15k/shards_std 对同一 id 给的是**不同字**,
则用前者训练、后者推理 = 训练/推理条件错配, 之前所有结论都要重估。
判据: 逐 id 的 latent 余弦 —— 同一张图应 >0.9; 不同字应 <0.5。
"""
import glob
import os
import sys

import numpy as np

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")

A = "data/top10_style23/shards_std"
B = "data/50k_v2_glyph15k/shards_std"
TGT = "data/top10_style23/shards_gtskel_w7"


def load(d, max_shards=1):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz")))[:max_shards]:
        try:
            with np.load(sp) as z:
                for j, i in enumerate(z["img_ids"]):
                    mp[int(i)] = np.asarray(z["latents"][j], np.float32)
        except Exception as e:                                   # noqa: BLE001
            print(f"  (读 {sp} 失败: {e})")
    return mp


for d in (A, B, TGT):
    print(f"{d}: {'存在' if os.path.isdir(d) else '不存在'}")

ma, mb = load(A), load(B)
print(f"\n{os.path.basename(A)}: {len(ma)} 条 | {os.path.basename(B)}: {len(mb)} 条")
common = sorted(set(ma) & set(mb))
print(f"共同 id: {len(common)}")


def cos(x, y):
    x, y = np.asarray(x, np.float32).ravel(), np.asarray(y, np.float32).ravel()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


if common:
    cs = [cos(ma[i], mb[i]) for i in common[:200]]
    print(f"逐 id 余弦 (前 {len(cs)} 个共同 id): 中位 {np.median(cs):+.4f}  "
          f"最小 {min(cs):+.4f} 最大 {max(cs):+.4f}")
    same = sum(1 for c in cs if c > 0.9)
    print(f"  >0.9 的条数: {same}/{len(cs)}  -> "
          f"{'同一命名空间 ✓' if same > 0.8 * len(cs) else '★ 命名空间不同, 会错配!'}")

# 若两套不同, 再各自与 GT(w7) 比, 看哪一套才是"与 GT 同字"的正确配对
mg = load(TGT)
if mg:
    for nm, m in (("top10/style23", ma), ("50k_v2_glyph15k", mb)):
        cc = sorted(set(m) & set(mg))
        if not cc:
            continue
        v = [cos(m[i], mg[i]) for i in cc[:200]]
        print(f"  {nm:16} vs GT(w7): 共同 {len(cc)}  余弦中位 {np.median(v):+.4f}")
