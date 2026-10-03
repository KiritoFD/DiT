#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_progress.py — 长训进度体检：分桶 Diff（看走平还是震荡）+ eval 轨迹 + poster 路径。

判读要点：
  * Diff 每桶均值仍在降 -> 还没到平台，继续跑；
  * 桶均值平了但**桶内方差大**（逐步 min/max 差很多）-> lr 偏高，在极小值附近震荡；
  * 桶均值平且桶内方差小 -> 真的收敛了，剩下的差距是容量/数据问题，不是 lr。
"""
import glob
import os
import re
import sys

os.chdir("/root/Workspace/xy/DiT")
RE_DIFF = re.compile(r"\(step=(\d+)\).*?Diff: ([0-9.]+)")
RE_EVAL = re.compile(r"step=(\d+) set=(\w+) n=(\d+) ssim=([\d.]+) \(med=([\d.]+)\).*?lpips=([\d.]+)")
BUCKET = 2000

logs = sys.argv[1:] or sorted(glob.glob("logs/v15_series/v15_fs6/*_long_*.log"),
                              key=os.path.getmtime)[-3:]
for lg in logs:
    diffs, evs = [], []
    for line in open(lg, encoding="utf-8", errors="replace"):
        m = RE_DIFF.search(line)
        if m:
            diffs.append((int(m.group(1)), float(m.group(2))))
        m = RE_EVAL.search(line)
        if m:
            evs.append((int(m.group(1)), float(m.group(4)), float(m.group(5)),
                        float(m.group(6))))
    name = os.path.basename(lg).split("_0921-")[0]
    print(f"\n===== {name}   ({len(diffs)} loss 行, {len(evs)} 次 eval)")
    if not diffs:
        print("   还没开始出 loss")
        continue
    base = diffs[0][0]
    buckets = {}
    for s, v in diffs:
        buckets.setdefault((s - base) // BUCKET, []).append(v)
    print("   Diff 分桶(每 2000 步): 均值 / 桶内min..max")
    for b in sorted(buckets)[:14]:
        vs = buckets[b]
        print(f"     +{b*BUCKET:>6d} 步  {sum(vs)/len(vs):.4f}   "
              f"{min(vs):.3f}..{max(vs):.3f}   n={len(vs)}")
    if evs:
        print("   eval ssim: " + "  ".join(f"{s-base}:{v:.4f}" for s, v, _, _ in evs))
        print(f"   最新 eval: step={evs[-1][0]} ssim={evs[-1][1]:.4f} "
              f"med={evs[-1][2]:.4f} lpips={evs[-1][3]:.4f}")

print("\n===== poster 位置")
for p in sorted(glob.glob("assets/results/v15_fs6_*long*/**/posters/*.png",
                          recursive=True), key=os.path.getmtime)[-12:]:
    print("   ", p, f"({os.path.getsize(p)//1024}K)")
