#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_stdskel_gap.py — 量化 std 骨架 latent 与 GT 结构的 latent 域 gap (CPU, 只读数据).

对 fame3 训练集前 N 个 img_id, 读三份 latent 两两比较 cosine 相似度:
  A = data/skel/std_skel1_latents_fame3_v8    (std 骨架, 训练 g)
  B = data/skel/final_skel_latents_fame_1px_v8 (GT 1px 实例骨架, 历史成功条件)
  C = data/latents/final_latents_fame3_v8         (GT 目标 latent, 训练目标 x0)

用法: python tools/debug_stdskel_gap.py --n 3000
"""
import os, sys, re, json, csv, glob, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)

import numpy as np


def load_all(shard_dir):
    """一次加载整个目录 -> dict {img_id: latent float32}. (~0.45G/目录, 可接受)"""
    d = {}
    for sp in sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz"))):
        with np.load(sp) as z:
            ids = np.asarray(z["img_ids"])
            lats = np.asarray(z["latents"], dtype=np.float32)
            for j in range(len(ids)):
                d[int(ids[j])] = lats[j]
    return d


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--csv", default="assets/train_fame3_clean_v8.csv")
    ap.add_argument("--out", default="assets/results/stdskel_gap.json")
    args = ap.parse_args()

    dirA, dirB, dirC = ("data/skel/std_skel1_latents_fame3_v8",
                        "data/skel/final_skel_latents_fame_1px_v8",
                        "data/latents/final_latents_fame3_v8")
    print("加载 A(std skel)...", flush=True); A = load_all(dirA)
    print("加载 B(GT 1px skel)...", flush=True); B = load_all(dirB)
    print("加载 C(GT target)...", flush=True); C = load_all(dirC)
    print(f"A={len(A)} B={len(B)} C={len(C)}", flush=True)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))[:args.n]
    ids = []
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            ids.append(int(m.group(1)))

    AB, AC, BC, nA, nB, nC = [], [], [], [], [], []
    for iid in ids:
        if iid not in A or iid not in B or iid not in C:
            continue
        a, b, c = A[iid], B[iid], C[iid]
        AB.append(cos(a, b)); AC.append(cos(a, c)); BC.append(cos(b, c))
        nA.append(np.linalg.norm(a.reshape(-1)))
        nB.append(np.linalg.norm(b.reshape(-1)))
        nC.append(np.linalg.norm(c.reshape(-1)))

    res = {"n": len(AB)}
    print(f"\n配对样本 {len(AB)}/{len(ids)}\n", flush=True)
    for label, arr in [("cos(A=std, B=GT_skel)", AB),
                       ("cos(A=std, C=GT_target)", AC),
                       ("cos(B=GT_skel, C=GT_target)", BC)]:
        a = np.array(arr); a = a[~np.isnan(a)]
        m, med = float(np.mean(a)), float(np.median(a))
        print(f"{label:<32} mean={m:+.4f}  med={med:+.4f}  "
              f"p10={np.percentile(a,10):+.4f}  p90={np.percentile(a,90):+.4f}", flush=True)
        res[label] = {"mean": round(m, 4), "median": round(med, 4)}

    print("\n能量 (L2 norm):", flush=True)
    for label, arr in [("|data/skel/std_skel|", nA), ("|GT_skel|", nB), ("|GT_target|", nC)]:
        m = float(np.mean(arr))
        print(f"  {label:<14} mean={m:.3f}", flush=True)
        res[label] = round(m, 3)

    res["nmse_AB"] = round(float(2 * (1 - np.nanmean(AB))), 4)
    res["nmse_AC"] = round(float(2 * (1 - np.nanmean(AC))), 4)
    res["nmse_BC"] = round(float(2 * (1 - np.nanmean(BC))), 4)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()