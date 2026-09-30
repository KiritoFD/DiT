# -*- coding: utf-8 -*-
"""diag_two_stage.py — 聚合两阶段实验 (gtskel / skelnet) 的逐样本 eval 指标.

只读。从 assets/results/<exp>/eval_stdskel_batch.csv 聚合,
输出每 (step, set) 的: ssim / ink_ssim / ink_iou / skel_iou / frag_ratio / hole_pred / hole_gt。

skel_iou  = 生成图的骨架 vs GT 骨架 (结构正确性)
ink_iou   = 墨迹 IoU
frag_ratio= 断笔率
hole_*    = 空洞 (预测/GT)
"""
import csv
import glob
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS = ["ssim", "ink_ssim", "ink_iou", "skel_iou", "frag_ratio",
           "hole_pred", "hole_gt"]
EXPS = sys.argv[1:] or ["v25_stdskel", "v26_gtskel", "v27_skelnet_front",
                        "v29_skelnet_v2", "v28_reg"]

for exp in EXPS:
    p = f"assets/results/{exp}/eval_stdskel_batch.csv"
    if not os.path.exists(p):
        print(f"\n### {exp}: 无 eval_stdskel_batch.csv")
        continue
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(p, encoding="utf-8")):
        key = (int(r["step"]), r["set"])
        for m in METRICS:
            try:
                acc[key][m].append(float(r.get(m, "") or "nan"))
            except Exception:
                pass
    rows = sorted(acc)
    if not rows:
        print(f"\n### {exp}: 空")
        continue
    print(f"\n### {exp}   (共 {len(rows)} 个 eval 点)")
    hdr = f"{'step':>7} {'set':>12} " + " ".join(f"{m:>10}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for k in rows:
        vals = []
        for m in METRICS:
            v = [x for x in acc[k][m] if x == x]
            vals.append(sum(v) / len(v) if v else float("nan"))
        print(f"{k[0]:7d} {k[1]:>12} " + " ".join(f"{x:10.4f}" for x in vals))
    # 总结: strict 集的最佳 skel_iou / ssim
    strict = [k for k in rows if "strict" in k[1]]
    if strict:
        best_i = max(strict, key=lambda k: (sum(acc[k]["skel_iou"]) / len(acc[k]["skel_iou"])))
        best_s = max(strict, key=lambda k: (sum(acc[k]["ssim"]) / len(acc[k]["ssim"])))
        bi = sum(acc[best_i]["skel_iou"]) / len(acc[best_i]["skel_iou"])
        bs = sum(acc[best_s]["ssim"]) / len(acc[best_s]["ssim"])
        print(f"  → strict 最佳 skel_iou {bi:.4f} @{best_i[0]} ({best_i[1]}) | "
              f"最佳 ssim {bs:.4f} @{best_s[0]} ({best_s[1]})")
