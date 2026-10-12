#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_sweep_images.py — 用"与 GT 墨量向量比对"区分 [样本组] 与 [GT组]，再正确落盘

step0000000/genrows/ 有 20 张 = 样本 10 + GT 10。先用 filename 排序分成前/后两组，
分别与 eval/gt 的墨量向量比对：**更接近 GT 的那组是 GT**，另一组是模型样本。
"""
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

os.chdir("/home/ds/Workspace/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEPS = ["0020000", "0035000", "0040000", "0050000", "0055000", "0100000", "0110000", "0115000"]
IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]


def ink(p):
    try:
        return float((np.asarray(Image.open(p).convert("L")) < 128).mean())
    except Exception:
        return None


# 由 eval200_fixed.csv 现取 GT（48 上有原图）
import csv  # noqa: E402
rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))
gt_ink = []
for i in IDX:
    p = rows[i]["image_path"]
    gt_ink.append(ink(p) if os.path.exists(p) else np.nan)
gt_ink = np.array(gt_ink)
print("  GT 墨量向量:", np.round(gt_ink, 3))

for st in STEPS:
    root = f"_local_ckpt/gen_v81sw_{st}_n10/eval_samples_ctrl/step0000000/genrows"
    if not os.path.isdir(root):
        print(f"  {st}: 缺 {root}")
        continue
    ps = sorted(glob.glob(os.path.join(root, "*.png")))
    inks = np.array([ink(p) for p in ps])
    n = len(ps)
    print(f"\n=== {st} ===  {n} 张: " + " ".join(os.path.basename(p) for p in ps[:6]) + " ...")
    # 按文件名把候选分两组：连续的 10+10 或 交错；这里穷举所有 10 张子集不现实,
    # 用"名字前缀"聚类：取文件名去掉数字后的前缀
    import re
    pref = {}
    for p in ps:
        k = re.sub(r"\d+", "#", os.path.basename(p))
        pref.setdefault(k, []).append(p)
    best = None
    for k, group in pref.items():
        if len(group) != len(IDX):
            continue
        gi = np.array([ink(p) for p in group])
        d_gt = float(np.nanmean(np.abs(gi - gt_ink)))
        print(f"   前缀 {k!r}: {len(group)} 张 | 与 GT 的 L1 距离={d_gt:.4f}")
        if best is None or d_gt > best[0]:
            best = (d_gt, k, group)      # 距离 **更大** 的 = 模型样本（越小越像 GT）
    if best is None:
        # 没有干净前缀：按排序切前/后各 10
        half = n // 2
        cands = [("first", ps[:half]), ("last", ps[half:])]
        scored = []
        for k, group in cands:
            gi = np.array([ink(p) for p in group])
            scored.append((float(np.nanmean(np.abs(gi - gt_ink))), k, group))
        scored.sort()
        print("    无干净前缀 -> 前/后各半: " + ", ".join(f"{k}:{d:.4f}" for d, k, _ in scored))
        best = (scored[-1][0], scored[-1][1], scored[-1][2])   # 取离 GT 更远者
    d_gt, k, group = best
    group = sorted(group)
    dst = f"eval/v81fix_{st}"
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    for i, p in enumerate(group):
        shutil.copyfile(p, os.path.join(dst, f"{i}.png"))
    got = [ink(os.path.join(dst, f"{i}.png")) for i in range(len(group))]
    print(f"    -> {dst}  用组 {k!r} (与GT距离 {d_gt:.4f})")
    print("       样本墨量:", " ".join(f"{v:.3f}" for v in got))

DL = " ".join(f"v81fix_{s}" for s in STEPS if os.path.isdir(f"eval/v81fix_{s}"))
os.system(f"rm -f /tmp/v81fix.tgz && tar -czf /tmp/v81fix.tgz -C eval {DL} && ls -la /tmp/v81fix.tgz")
