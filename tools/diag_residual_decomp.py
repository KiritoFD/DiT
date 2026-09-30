# -*- coding: utf-8 -*-
"""diag_residual_decomp.py — 形变残差 r = x0(GT骨架) − g(标准骨架) 到底有多少是可解释的。

目的: 判断 SkelNet 学不会是 (a) 容量/优化 还是 (b) **条件里根本没有这个信息**。

做法 (纯 numpy, 不训练):
  对每个样本算 r = x0 - g, 然后按「风格槽位(书家×书体)」分组做方差分解:
    · 总方差 Var(r)
    · 槽位间方差 Var_between(槽位均值残差)
    · R²_style = Var_between / Var_total     <- 风格能解释多少
  若 R²_style 很低, 说明残差主要是**逐字特异**的, 只给(标准骨架 + 风格向量)
  在原理上就推不出来 -> 不是容量问题, 是条件信息不足。

同时也给出:
  · cos(x0, g)   —— 目标与输入的固有重合度 (决定"照抄"能拿多少分)
  · 每个槽位/字的样本数, 以及同一 (槽位,字) 是否有多张 (决定能否学"逐字"结构)
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


def load_map(d, limit=None):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            lat = z["latents"]
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = np.asarray(lat[j], dtype=np.float32)
        if limit and len(mp) >= limit:
            break
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_top10_style23.csv")
    ap.add_argument("--tgt", default="data/top10_style23/shards_gtskel_w3")
    ap.add_argument("--cond", default="data/top10_style23/shards_std")
    ap.add_argument("--map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"csv {len(rows)} 行, 列={list(rows[0].keys())[:14]}")
    csmap = json.load(open(a.map, encoding="utf-8")) if os.path.exists(a.map) else {}
    print(f"style map: num_pairs={csmap.get('num_pairs')} "
          f"num_calligraphers={csmap.get('num_calligraphers')}")

    def slot_of(r):
        # 优先用 csv 自带的 slot_id; 否则用 (书家,书体) 名字映射
        if "slot_id" in r and str(r["slot_id"]).strip():
            return int(r["slot_id"])
        key = f"{r['calligrapher']}_{r['script']}"
        return csmap.get("pair_to_id", {}).get(key, hash(key) % 1000)

    tgt = load_map(a.tgt)
    cond = load_map(a.cond)
    print(f"tgt {len(tgt)} ids | cond {len(cond)} ids")

    Rs, slots, chars, cos_xg = [], [], [], []
    by_slot = defaultdict(list)
    by_pair = defaultdict(int)
    used = 0
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if not m:
            continue
        i = int(m.group(1))
        if i not in tgt or i not in cond:
            continue
        x0 = tgt[i].astype(np.float64).ravel()
        g = cond[i].astype(np.float64).ravel()
        Rs.append(x0 - g)
        slots.append(slot_of(r))
        chars.append(r.get("character", "?"))
        cx = float((x0 @ g) / (np.linalg.norm(x0) * np.linalg.norm(g) + 1e-12))
        cos_xg.append(cx)
        by_slot[slot_of(r)].append(x0 - g)
        by_pair[(slot_of(r), r.get("character", "?"))] += 1
        used += 1
        if used >= a.n:
            break
    print(f"\n有效样本 {used}")

    R = np.stack(Rs)
    S = np.array(slots)
    print(f"\n[1] 目标与输入的固有重合度")
    print(f"    cos(x0, g) = {np.mean(cos_xg):.4f}   <- 越高说明'照抄'越接近答案")
    print(f"    残差能量占比 ≈ 1 - cos² = {1 - np.mean(cos_xg)**2:.3f}")

    print(f"\n[2] 残差的风格可解释性 (按槽位做方差分解)")
    tot = R.var(axis=0).sum() + (R.mean(axis=0) ** 2).sum() * 0  # 总方差(含均值)
    grand = R.mean(axis=0)
    tot_ss = float(((R - grand) ** 2).sum())
    between_ss = 0.0
    for s, rs in by_slot.items():
        rs = np.stack(rs)
        between_ss += float(len(rs) * ((rs.mean(axis=0) - grand) ** 2).sum())
    r2 = between_ss / max(tot_ss, 1e-12)
    print(f"    槽位数 {len(by_slot)}, 每槽位样本数 "
          f"min={min(len(v) for v in by_slot.values())} "
          f"max={max(len(v) for v in by_slot.values())}")
    print(f"    ★ R²_style = {r2*100:.2f}%   <- 风格槽位能解释的残差方差")
    print(f"    -> 其余 {100 - r2*100:.2f}% 是**同槽位内的逐字差异**, 只给风格向量推不出来")

    print(f"\n[3] 同一 (槽位,字) 是否有多张 (决定能否学逐字结构)")
    mult = sum(1 for v in by_pair.values() if v > 1)
    print(f"    (槽位,字) 组合数 {len(by_pair)}, 其中样本数>1 的 {mult} "
          f"({mult/max(len(by_pair),1)*100:.1f}%)")
    if by_pair:
        mx = max(by_pair.values())
        print(f"    单组合最大样本数 = {mx}")


if __name__ == "__main__":
    main()
