#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_evals.py — 把若干模型的 eval_auto_*.json 导出为 CSV（name,step,ssim,lpips,frag,ink）

用法: python dump_evals.py "v68=assets/runs/v68_family/v68_aug_sp_c2ot" "noaug_c2ot=assets/results/ablation/noaug_c2ot"
对每个 name=dir, 在 dir 下**递归**找 checkpoints/eval_auto_*.json, 取 set==eval200fix 的。
"""
import glob
import json
import os
import sys

rows = []
for spec in sys.argv[1:]:
    name, base = spec.split("=", 1)
    best = {}
    for p in glob.glob(os.path.join(base, "**", "eval_auto_*.json"), recursive=True):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(j, dict):
            continue
        if j.get("set") not in ("eval200fix", "eval200_fixed"):
            continue
        try:
            st = int(j.get("step"))
        except Exception:
            continue
        # 同 step 取文件更新的（多 run 目录时）
        if st not in best or os.path.getmtime(p) > best[st][0]:
            best[st] = (os.path.getmtime(p), j)
    for st in sorted(best):
        j = best[st][1]
        rows.append((name, st, j.get("ssim"), j.get("lpips"), j.get("frag"), j.get("ink_iou")))
print("name,step,ssim,lpips,frag,ink")
for r in sorted(rows, key=lambda x: (x[0], x[1])):
    print(",".join("" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v)) for v in r))
