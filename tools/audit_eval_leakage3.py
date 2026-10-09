# -*- coding: utf-8 -*-
"""audit_eval_leakage3.py — 收尾审计:
  A. per-sample CSV 里有哪些模型 (决定'组合见过优势'是不是 v68 独有 => 记忆 还是 难度);
  B. 项目自带的严格子集 assets/eval_top10_strict_subset84.csv 是什么、与训练的字/组合重叠如何;
  C. 产出"组合未见"的 112 个样本清单, 供需要严格口径时重做海报。
"""
import csv
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL = "exp-std/csv/eval200_fixed.csv"
TRAIN = "exp-std/csv/train.csv"
PER_SAMPLE = "assets/eval200fix_models_per_sample.csv"


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("=== A. per-sample CSV 里的模型 ===")
    ps = rows(PER_SAMPLE)
    print("  ", Counter(r["model_name"] for r in ps))

    ev = rows(EVAL)
    tr = rows(TRAIN)
    K3 = ("calligrapher", "script", "character")
    seen3 = {tuple(str(r.get(k, "")).strip() for k in K3) for r in tr}

    # 每个模型: 组合见过 vs 未见 的均值 SSIM (全 187)
    print("\n=== B. 各模型的 组合见过/未见 均值 SSIM (全 187) ===")
    idx_seen = {}
    for i, r in enumerate(ev):
        idx_seen[i] = tuple(str(r.get(k, "")).strip() for k in K3) in seen3
    by_model = {}
    for r in ps:
        by_model.setdefault(r["model_name"], []).append((int(r["idx"]), float(r["ssim"])))
    print(f"  {'model':12s} {'见过SSIM':>9s} {'未见SSIM':>9s} {'Δ':>8s}  (见过 n / 未见 n)")
    for m, lst in by_model.items():
        a = [s for i, s in lst if idx_seen.get(i)]
        b = [s for i, s in lst if not idx_seen.get(i)]
        if not a or not b:
            print(f"  {m:12s} 数据不足 (n={len(lst)})")
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        print(f"  {m:12s} {ma:9.4f} {mb:9.4f} {ma-mb:+8.4f}  ({len(a)} / {len(b)})")

    print("\n=== C. 项目自带的严格子集 ===")
    for cand in ("assets/eval_top10_strict_subset84.csv",
                 "assets/eval_top10_strict_subset84_fixed.csv",
                 "exp-std/csv/eval_top10_strict_subset84.csv"):
        if os.path.isfile(cand):
            rs = rows(cand)
            chars = {str(r.get("character", "")).strip() for r in rs}
            tr_chars = {str(r.get("character", "")).strip() for r in tr}
            print(f"  {cand}: {len(rs)} 行, {len(chars)} 个不同字, "
                  f"其中在训练字表里 {len(chars & tr_chars)} 个")
            print(f"    列: {list(rs[0].keys())[:10]}")
        else:
            print(f"  (无 {cand})")

    # 写"组合未见"清单
    keep = [r for i, r in enumerate(ev) if not idx_seen[i]]
    out = "exp-std/csv/eval200_combo_unseen.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev[0].keys()))
        w.writeheader()
        w.writerows(keep)
    print(f"\n=== D. 已写 {out}: {len(keep)} 行 (组合未见子集, 需要严格口径时用) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
