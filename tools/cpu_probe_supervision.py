"""CPU-only 诊断: 对 latent skel, 什么监督信号才够好?

核心量 = **条件不可约方差** σ²_irr:
  同一条件(标准字骨架 + 书家)下, 目标(真迹骨架)仍然散开的那部分方差。
  L2/flow 的最优解就是条件均值 -> σ²_irr 越大, 学出来的越糊(均值解)。
  所以"好监督" = 让 σ²_irr 尽量小。

本脚本在 latent 空间(纯 numpy, 不动 VAE/GPU)对候选信号排序:
  A. 原始真迹骨架 latent              -> σ²_raw
  B. 形变残差 (gt - std)              -> σ²_delta   (把条件本身扣掉)
  C. OT/最近邻指派后的目标            -> σ²_ot     (一条件只配最像它的那条真迹)
  D. 逐通道白化后的 A/B/C
  E. 组均值(medoid)                   -> 下界参考

另附: 3px vs 7px 在 latent 上的可观测性 (支持度/高频能量)。
"""
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

CSV = "assets/train_top10_style23_real.csv"
GT = "data/top10_style23/shards_gtskel_w7"     # 真迹骨架 (目标)
STD_W7 = "data/top10_style23/shards_std_w7"    # 标准字骨架 w7 (条件)
STD_3 = "data/top10_style23/shards_std"        # 标准字骨架 3px (对照)
CAP = 12000


def _read_shard(task):
    """worker: 解压一个分片, 只回传需要的 id (避免整包跨进程传输)。"""
    f, want = task
    out = {}
    with np.load(f) as z:
        lat = z["latents"]
        for j, i in enumerate(z["img_ids"]):
            i = int(i)
            if want is None or i in want:
                out[i] = np.asarray(lat[j], np.float32)
    return out


def load(dd, want=None, workers=16):
    """★ 多进程按分片并行 (savez_compressed 每次都要整包解压, 串行会非常慢)。"""
    import multiprocessing as mp
    fs = sorted(glob.glob(os.path.join(dd, "shard_*.npz")))
    m = {}
    with mp.Pool(min(workers, max(len(fs), 1))) as p:
        for d in p.imap_unordered(_read_shard, [(f, want) for f in fs]):
            m.update(d)
    return m


rows = []
with open(CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        mm = re.search(r"(\d+)\.png$", r.get("image_path", "") or "")
        if mm:
            rows.append((int(mm.group(1)),
                         str(r.get("character", "") or r.get("char", "")),
                         str(r.get("calligrapher", "") or r.get("calligrapher_id", ""))))
print(f"[csv] {len(rows)} 行, 列样本: {rows[0]}")
ids_all = {r[0] for r in rows}

gt = load(GT, ids_all)
std = load(STD_W7, ids_all)
common = sorted(set(gt) & set(std))
print(f"[load] gt={len(gt)} std={len(std)} 交集={len(common)}")
if len(common) > CAP:
    common = common[:CAP]

X = np.stack([std[i] for i in common])       # 条件
Y = np.stack([gt[i] for i in common])        # 目标
meta = {r[0]: (r[1], r[2]) for r in rows}
print(f"[data] X{Y.shape}  总方差 var(Y)={Y.var():.4f}")

# ── 分组: 同一 (字, 书家) 视为同一条件 ──
groups = {}
for k, i in enumerate(common):
    ch, ca = meta.get(i, ("?", "?"))
    groups.setdefault((ch, ca), []).append(k)
sizes = np.array([len(v) for v in groups.values()])
print(f"[group] {len(groups)} 组, 平均 {sizes.mean():.2f} 条/组, "
      f"多值组(>=2) 占比 {(sizes >= 2).mean():.2%}")


def conditional_var(sig, name):
    """条件不可约方差: 组内方差的均值 (按样本加权)。"""
    tot, n = 0.0, 0
    for g in groups.values():
        arr = sig[g]
        tot += arr.var(axis=0, ddof=0).sum() * len(g)
        n += len(g)
    v = tot / max(n, 1)
    print(f"  {name:<34} 条件不可约方差 = {v:.4f}   "
          f"(占总方差 {v / max(sig.var() * sig[0].size, 1e-9):.2%})")
    return v


print("\n=== 候选监督信号: 条件不可约方差 (越小越好) ===")
v_raw = conditional_var(Y, "A. 原始真迹骨架 latent")
delta = Y - X
v_delta = conditional_var(delta, "B. 形变残差 (gt - std)")

# 逐通道白化 (用训练集统计)
w = 1.0 / (Y.reshape(len(Y), Y.shape[1], -1).std(axis=(0, 2)).reshape(1, -1, 1, 1) + 1e-6)
v_white = conditional_var(Y * w, "D1. A + 逐通道白化")
v_dwhite = conditional_var(delta * w, "D2. B + 逐通道白化")

# C. OT / 最近邻指派: 每个条件只配"最像它的那条真迹" -> 取组内 medoid 作为目标
Y_ot = Y.copy()
for g in groups.values():
    arr = Y[g]
    if len(arr) <= 1:
        continue
    d = ((arr[:, None] - arr[None]).reshape(len(arr), len(arr), -1) ** 2).sum(-1)
    med = arr[int(np.argmin(d.sum(1)))]
    Y_ot[g] = med
v_ot = conditional_var(Y_ot, "C. OT/最近邻指派后的目标")

# E. 组均值 (L2 的理论最优解, 下界参考)
Y_mean = Y.copy()
for g in groups.values():
    if len(g) > 1:
        Y_mean[g] = Y[g].mean(0)
v_mean = conditional_var(Y_mean, "E. 组均值 (L2 会收敛到的解)")

print("\n=== 3px vs 7px 在 latent 上的可观测性 ===")
for nm, dd in (("3px std", STD_3), ("7px std", STD_W7)):
    s3 = load(dd, set(common[:2000]))
    if not s3:
        print(f"  {nm}: 无数据")
        continue
    A = np.stack(list(s3.values())[:2000])
    gx = np.abs(np.diff(A, axis=2)).mean()
    gy = np.abs(np.diff(A, axis=3)).mean()
    supp = float((np.abs(A) > 0.5).mean())
    print(f"  {nm:<8} |z|>0.5 支持度={supp:.4f}  高频能量(|∇z|)={0.5*(gx+gy):.4f}  "
          f"std={A.std():.4f}")

print("\n判读: 哪个候选的'条件不可约方差'最小, 就最不容易被 L2 逼成糊的均值解; "
      "若 C(OT) 相比 A 大幅下降, 说明多值性确实是可修的。")
