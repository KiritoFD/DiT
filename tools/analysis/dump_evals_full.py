# -*- coding: utf-8 -*-
"""dump_evals_full.py — 导出 eval_auto_*.json 的**全部数值字段**（不只是一行 4 个指标）

用法: python dump_evals_full.py "A0=assets/results/v83_v68_noREPA" "v68=assets/runs/v68_family/v68_aug_sp_c2ot"
输出: CSV 到 stdout（首行表头）
"""
import glob
import json
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_R if os.path.isdir(os.path.join(_R, "src")) else os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIELD_ORDER = ["ssim", "lpips", "mse", "frag", "ink_iou", "ink_ssim", "skel_iou",
               "hole", "hole_gt", "nn", "tgt_spec", "cal_enrich", "psnr", "hf"]
rows = []
keys_seen = set()
for spec in sys.argv[1:]:
    name, base = spec.split("=", 1)
    best = {}
    for p in glob.glob(os.path.join(base, "**", "eval_auto_*.json"), recursive=True):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(j, dict) or j.get("set") not in ("eval200fix", "eval200_fixed"):
            continue
        try:
            st = int(j.get("step"))
        except Exception:
            continue
        if st not in best or os.path.getmtime(p) > best[st][0]:
            best[st] = (os.path.getmtime(p), j)
    for st in sorted(best):
        j = best[st][1]
        r = {"name": name, "step": st}
        for k, v in j.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "step":
                r[k] = v
                keys_seen.add(k)
        rows.append(r)

cols = ["name", "step"] + [k for k in FIELD_ORDER if k in keys_seen] + \
       sorted(k for k in keys_seen if k not in FIELD_ORDER)
print(",".join(cols))
for r in sorted(rows, key=lambda x: (x["name"], x["step"])):
    out = []
    for c in cols:
        v = r.get(c)
        out.append("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
    print(",".join(out))
