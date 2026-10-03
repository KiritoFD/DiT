"""数据质量审计 (纯 CPU, 不碰 GPU):
  ① 反色/碑文样本: 目标图是"白字黑底"(拓片) -> 与训练目标(黑字白底)相反, 会污染监督。
     判据: 目标 PNG 的平均亮度 < 0.5 (正常是 ~0.9)。
  ② 条件/目标字不一致 (繁简): std 标准字骨架来自简体字形, 而 GT 是繁体真迹 ->
     条件与目标不是同一个字。判据: std 图与 GT 图在**墨迹域**的一致度 (skel_iou / ink_iou)
     落到分布尾部的那批, 人工看几张即可定性。

输出: 反色样本数与 img_id 样例; eval200 里条件-目标最不一致的 10 条; 分槽位统计。
"""
import csv
import os
import re
import sys

import numpy as np
from PIL import Image
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

from src.eval.metrics import skel_iou  # noqa: E402
from src.eval.metrics_ink import ink_iou  # noqa: E402

TRAIN_CSV = "exp-std/csv/train.csv"
IMG_ROOT = "data/top10_style23"
CACHE = "exp-std/data/eval_real200_cache.pt"


def lum(p):
    try:
        im = Image.open(p).convert("L").resize((48, 48))
        return float(np.asarray(im, dtype=np.float32).mean() / 255.0)
    except Exception:                                        # noqa: BLE001
        return None


# ── ① 训练集: 反色扫描 ───────────────────────────────────────────────────
with open(TRAIN_CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"[train] {TRAIN_CSV} n={len(rows)}")

vals, dark = [], []
for r in rows:
    p = os.path.join(IMG_ROOT, "imgs", os.path.basename(r["image_path"]))
    v = lum(p)
    if v is None:
        continue
    vals.append(v)
    if v < 0.5:
        dark.append((int(r["img_id"]), round(v, 3), r.get("slot_name"), r.get("source")))
vals = np.array(vals)
print(f"[① 反色] 扫描 {len(vals)} 张: 亮度<0.5 的 = {len(dark)} "
      f"({len(dark) / max(len(vals), 1):.2%})")
print(f"         亮度分布 p1/p10/中位/p90 = "
      f"{np.percentile(vals, 1):.3f}/{np.percentile(vals, 10):.3f}/"
      f"{np.median(vals):.3f}/{np.percentile(vals, 90):.3f}")
for d in sorted(dark, key=lambda x: x[1])[:10]:
    print(f"         id={d[0]} lum={d[1]} slot={d[2]} src={d[3]}")
from collections import Counter  # noqa: E402

print(f"         反色按来源: {dict(Counter(d[3] for d in dark).most_common(6))}")

# ── ② eval200: 条件(std) 与 目标(GT) 的字一致性 ──────────────────────────
c = th.load(CACHE, map_location="cpu", weights_only=False)
gt = c["gt_pngs"].numpy().astype(np.float32)
std = c["std_pngs"].numpy().astype(np.float32)
ids = [int(i) for i in c["img_ids"]]
srows = c["rows"]
per = []
for i in range(len(ids)):
    per.append(dict(
        id=ids[i], slot=srows[i].get("slot_name"), src=srows[i].get("source"),
        gt_lum=float(gt[i].mean()), std_lum=float(std[i].mean()),
        sk=float(skel_iou(std[i], gt[i], thresh=0.5)),
        ink=float(ink_iou(std[i], gt[i])),
    ))
sk = np.array([p["sk"] for p in per])
print(f"\n[② 条件-目标一致性] eval200: skel_iou(std,GT) 中位={np.median(sk):.3f} "
      f"p10={np.percentile(sk, 10):.3f} p90={np.percentile(sk, 90):.3f}")
invg = [p for p in per if p["std_lum"] < 0.5 or p["gt_lum"] < 0.5]
print(f"         eval200 里反色的 = {len(invg)} (std 或 GT 亮度<0.5)")
worst = sorted(per, key=lambda p: p["sk"])[:10]
print("         最不一致的 10 条 (疑似繁简/反色):")
for p in worst:
    print(f"         id={p['id']} skel_iou={p['sk']:.3f} ink_iou={p['ink']:.3f} "
          f"gt_lum={p['gt_lum']:.3f} std_lum={p['std_lum']:.3f} "
          f"slot={p['slot']} src={p['src']}")
n_bad = int((sk < 0.10).sum())
print(f"         skel_iou<0.10 的 = {n_bad}/200 ({n_bad/200:.1%})  <- 条件与目标基本不同字")
