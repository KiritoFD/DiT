"""训练集"条件与目标不同字"比例 —— **图像域判据** (纯 CPU)。

判据与 eval200 上完全一致: skel_iou(条件 std 图, 目标真迹图)。eval200 上它是干净的
双峰 (好样本 1.000 / 坏样本 0.000), 远好于潜空间 L2 代理(已废弃)。
"""
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
from src.eval.metrics import skel_iou  # noqa: E402

TRAIN = "exp-std/csv/train.csv"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = 全量


def im(p):
    g = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    return np.repeat(g[..., None], 3, axis=2)   # skel_iou 要 HWC 3 通道


with open(TRAIN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if LIMIT:
    rows = rows[:LIMIT]
print(f"[in] {len(rows)} 条 (全量={not bool(LIMIT)})", flush=True)

t0 = time.time()
vals = []
per_slot = {}
miss = 0
for n, r in enumerate(rows):
    bid = os.path.basename(r["image_path"])
    ps = os.path.join("data/top10_style23/std", bid)
    pg = os.path.join("data/top10_style23/imgs", bid)
    if not (os.path.exists(ps) and os.path.exists(pg)):
        miss += 1
        continue
    v = float(skel_iou(im(ps), im(pg), thresh=0.5))
    vals.append(v)
    s = r.get("slot_name")
    t = per_slot.setdefault(s, [0, 0])
    t[0] += 1
    if v < 0.10:
        t[1] += 1
    if (n + 1) % 5000 == 0:
        print(f"   ... {n+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

v = np.array(vals)
print(f"\n[结果] 有效 {len(v)} 条, 缺文件 {miss}")
print(f"  skel_iou(std, GT真迹) 分布: 中位={np.median(v):.3f} "
      f"p10={np.percentile(v,10):.3f} p25={np.percentile(v,25):.3f}")
for t in (0.05, 0.10, 0.20):
    k = int((v < t).sum())
    print(f"  skel_iou < {t:.2f}: {k} ({k/len(v):.1%})")
print("[按槽位] 坏(<0.10)占比 前 12:")
for s, (n, b) in sorted(per_slot.items(), key=lambda x: -(x[1][1]/max(x[1][0],1)))[:12]:
    print(f"   {s:<14} {b:5d}/{n:<5d} = {b/max(n,1):6.1%}")
print(f"[耗时] {time.time()-t0:.0f}s")
