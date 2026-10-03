"""量"条件与目标不是同一个字"(繁简/异体) 在**训练集**里的比例 + 拉出可视证据。

思路: eval200 已知 39 条是坏的(图像域 skel_iou(std,GT)<0.10)。用同样的两条 shard
目录(std_w7 / gtskel_w7)算潜空间 L2, 拿 200 条标定一个阈值, 再把这个阈值套到训练集。
纯 CPU, 不碰 GPU。
"""
import csv
import glob
import os
import sys

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

STD = "exp-std/data/shards_std_w7"
GTS = "exp-std/data/shards_gtskel_w7"
CACHE = "exp-std/data/eval_real200_cache.pt"
TRAIN = "exp-std/csv/train.csv"
OUT = "_ot_scratch/audit_pairs"
os.makedirs(OUT, exist_ok=True)


def load_map(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            arr = np.asarray(z["latents"], np.float32)
            for j, i in enumerate(z["img_ids"]):
                m[int(i)] = arr[j]
    return m


print("[load] std_w7 / gtskel_w7 ...", flush=True)
A, B = load_map(STD), load_map(GTS)
print(f"[load] std={len(A)} gt_skel={len(B)}", flush=True)

c = th.load(CACHE, map_location="cpu", weights_only=False)
ev_ids = [int(i) for i in c["img_ids"]]
gt, std = c["gt_pngs"].numpy(), c["std_pngs"].numpy()


def lat_l2(i):
    x, y = A.get(i), B.get(i)
    if x is None or y is None:
        return None
    return float(np.mean((x - y) ** 2))


from src.eval.metrics import skel_iou  # noqa: E402

ev = []
for k, i in enumerate(ev_ids):
    d = lat_l2(i)
    if d is None:
        continue
    ev.append((i, d, float(skel_iou(std[k], gt[k], thresh=0.5))))
ev.sort(key=lambda x: x[1])
bad = [e for e in ev if e[2] < 0.10]
good = [e for e in ev if e[2] >= 0.50]
print(f"[校准] eval200: 坏 {len(bad)} 条 latentL2 中位={np.median([e[1] for e in bad]):.4f} | "
      f"好 {len(good)} 条 latentL2 中位={np.median([e[1] for e in good]):.4f}")
thr = float(np.percentile([e[1] for e in good], 95))
print(f"[校准] 阈值 = 好样本 L2 的 p95 = {thr:.4f}  "
      f"(坏样本中位 L2 = {np.median([e[1] for e in bad]):.4f})")

# ── 训练集 ───────────────────────────────────────────────────────────────
with open(TRAIN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
vals, per_slot = [], {}
for r in rows:
    i = int(r["img_id"])
    d = lat_l2(i)
    if d is None:
        continue
    vals.append((i, d, r.get("slot_name")))
    s = r.get("slot_name")
    t = per_slot.setdefault(s, [0, 0])
    t[0] += 1
    if d > thr:
        t[1] += 1
n_bad = sum(1 for v in vals if v[1] > thr)
print(f"\n[训练集] 可对齐 {len(vals)}/{len(rows)} 条 | latentL2 > 阈值 = "
      f"{n_bad} ({n_bad / max(len(vals), 1):.1%})  <- 条件与目标很可能不同字")
print("[训练集] 按槽位的坏样本占比 (前 12 高):")
for s, (n, b) in sorted(per_slot.items(), key=lambda x: -(x[1][1] / max(x[1][0], 1)))[:12]:
    print(f"   {s:<14} {b:5d}/{n:<5d} = {b / max(n, 1):6.1%}")

# ── 可视证据: 最差的 4 对 (std 条件 vs GT 目标) 拼一张条图 ────────────────
worst = [e[0] for e in ev[:4]]
row_imgs = []
for i in worst:
    k = ev_ids.index(i)
    a = Image.fromarray((std[k].transpose(1, 2, 0) * 255).astype(np.uint8)).resize((160, 160))
    b = Image.fromarray((gt[k].transpose(1, 2, 0) * 255).astype(np.uint8)).resize((160, 160))
    strip = Image.new("L", (320, 160), 255)
    strip.paste(a.convert("L"), (0, 0))
    strip.paste(b.convert("L"), (160, 0))
    row_imgs.append(strip)
canvas = Image.new("L", (320, 160 * len(row_imgs)), 255)
for j, im in enumerate(row_imgs):
    canvas.paste(im, (0, 160 * j))
p = os.path.join(OUT, "std_vs_gt_worst4.png")
canvas.save(p)
print(f"\n[图] 最差 4 对 (左=条件 std, 右=目标 GT): {p}  ids={worst}")
