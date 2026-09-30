#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_stdskel_consistency.py — 检查 top10 与 fame 两套 **std 骨架** 是否一致。

动机: 想"把 top10 SkelNet 的 predskel 喂进 fame ControlNet"，前提是两边喂给
SkelNet 的 std 骨架属于**同一渲染约定**；否则是分布外输入，结论不可信。

方法 (CPU, numpy only):
  1. 各自建 character -> std 骨架 latent 的映射 (std 骨架按字决定, 与书家无关);
  2. 取共同字, 比较 latent 的 cosine / 相对 L2;
  3. 对照组: 同线内**不同字**之间的 cosine (下界参考);
  4. 若"同字跨线" cosine ≈ 1 且远高于"异字" -> 两套 std 骨架同源, 可直接复用。
"""
import csv
import glob
import os
import sys

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

TOP10_SH = "data/top10_style23/shards_std"
TOP10_CSV = "assets/train_top10_style23.csv"
FAME_SH = "data/archive/legacy_skeletons/std_skel1_latents_fame"
FAME_CSV = "data/archive/legacy_csv/train_fame.csv"


def load_shards(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            ids, lat = z["img_ids"], z["latents"]
            for j, i in enumerate(ids):
                m[int(i)] = lat[j].astype(np.float32)
    return m


def char_map(shards, csv_path, id_col):
    """character -> latent (同一字取第一个出现的 id; std 骨架与书家无关)。"""
    out = {}
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        ch = r["character"]
        if ch in out:
            continue
        iid = int(r[id_col]) if id_col in r else None
        if iid is None:
            iid = int(os.path.basename(r["image_path"])[:-4])
        if iid in shards:
            out[ch] = shards[iid]
    return out


def cos(a, b):
    a = a.ravel()
    b = b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 78)
    print("[1] 载入两套 std 骨架")
    print("=" * 78)
    t10s = load_shards(TOP10_SH)
    fms = load_shards(FAME_SH)
    t10 = char_map(t10s, TOP10_CSV, "img_id")
    fm = char_map(fms, FAME_CSV, "img_id")
    print("  top10: shards=%d ids, 覆盖字=%d" % (len(t10s), len(t10)))
    print("  fame : shards=%d ids, 覆盖字=%d" % (len(fms), len(fm)))

    common = sorted(set(t10) & set(fm))
    print("  共同字 = %d" % len(common))
    if not common:
        print("  ✗ 无共同字, 无法比较"); return

    print()
    print("=" * 78)
    print("[2] 同字跨线 cosine (越高越说明同源)")
    print("=" * 78)
    sims = [cos(t10[c], fm[c]) for c in common]
    sims = np.array(sims)
    print("  n=%d  mean=%.4f  median=%.4f  min=%.4f  max=%.4f"
          % (len(sims), sims.mean(), np.median(sims), sims.min(), sims.max()))
    for thr in (0.99, 0.95, 0.90, 0.80, 0.50):
        print("    cos >= %.2f : %5.1f%%" % (thr, 100.0 * (sims >= thr).mean()))

    print()
    print("=" * 78)
    print("[3] 对照组: 同线内异字 cosine (下界)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    keys = list(t10)
    a = rng.choice(len(keys), 2000)
    b = rng.choice(len(keys), 2000)
    diffs = [cos(t10[keys[i]], t10[keys[j]]) for i, j in zip(a, b) if i != j]
    diffs = np.array(diffs)
    print("  top10 异字 cosine: mean=%.4f  median=%.4f" % (diffs.mean(), np.median(diffs)))

    print()
    print("=" * 78)
    print("[4] 判读")
    print("=" * 78)
    print("  同字跨线 mean=%.4f   异字 mean=%.4f   差=%.4f"
          % (sims.mean(), diffs.mean(), sims.mean() - diffs.mean()))
    if sims.mean() > 0.95:
        print("  => ✅ 两套 std 骨架**同源**, top10 SkelNet 可直接用于 fame 的 std 骨架")
    elif sims.mean() > 0.80:
        print("  => ⚠️ 高度相似但不完全相同: 存在轻微渲染差异, 需谨慎")
    else:
        print("  => ❌ 差异明显: 直接复用属于分布外, 结论不可信, 需重训 SkelNet")

    # 逐字细节: 打印最差/最好的几个
    order = np.argsort(sims)
    print()
    print("  最不一致 8 个: %s" % [(common[i], round(float(sims[i]), 3)) for i in order[:8]])
    print("  最一致   8 个: %s" % [(common[i], round(float(sims[i]), 3)) for i in order[-8:]])


if __name__ == "__main__":
    main()
