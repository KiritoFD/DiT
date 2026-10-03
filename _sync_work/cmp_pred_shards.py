#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比两条 pred 生成路径的 shards —— 定位 0.6427 vs 0.5399 的矛盾。

只比 latent 统计 (不解码): 若某一路的 mean 符号/量级明显不同, 就是极性或口径错了。
"""
import glob
import os
import sys

import numpy as np

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")

DIRS = [
    ("我的 w7raw (gen_predskel_dit)", "data/top10_style23/predskel_dit_strict84_w7raw"),
    ("文档 REDO (eval_union_ckpt 重跑)", "data/top10_style23/predskel_eval_strict84_REDO"),
    ("文档 TS (eval_union_ckpt 原版)", "data/top10_style23/predskel_eval_strict84_TS"),
    ("std 骨架 (输入, 50k命名空间)", "data/50k_v2_glyph15k/shards_std"),
    ("GT 骨架 eval_strict84", "data/top10_style23/gt_skel_eval_strict84"),
]


def load(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        try:
            with np.load(sp) as z:
                for j, i in enumerate(z["img_ids"]):
                    mp[int(i)] = np.asarray(z["latents"][j], dtype=np.float32)
        except Exception as e:                                  # noqa: BLE001
            print(f"    (读 {sp} 失败: {e})")
    return mp


ref = None
for nm, d in DIRS:
    if not os.path.isdir(d):
        print(f"{nm:34}: 目录不存在 ({d})")
        continue
    mp = load(d)
    if not mp:
        print(f"{nm:34}: 空 ({d})")
        continue
    A = np.stack(list(mp.values()))
    print(f"{nm:34}: n={len(mp):4d}  latent mean={A.mean():+.4f}  std={A.std():.4f}  "
          f"|lat|={np.linalg.norm(A.reshape(len(mp), -1), axis=1).mean():6.2f}")
    if ref is None:
        ref = mp
    else:
        common = sorted(set(ref) & set(mp))
        if common:
            a = np.stack([ref[i] for i in common]).reshape(len(common), -1)
            b = np.stack([mp[i] for i in common]).reshape(len(common), -1)
            c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            print(f"{'':34}  ↳ 与「我的 w7raw」重叠 {len(common)} 条, 逐元素相关 = {c:+.4f}")
