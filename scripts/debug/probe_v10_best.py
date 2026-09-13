#!/usr/bin/env python
"""probe_v10_best.py — v10a/v10b base SSIM 峰值对比 (eval_auto in checkpoints/)"""
import json, glob

for tag in ["v10a_skel_cond_pretrain", "v10b_skel_only_pretrain"]:
    fs = sorted(glob.glob("/root/Workspace/xy/DiT/assets/results/%s/*/checkpoints/eval_auto_*.json" % tag))
    if not fs:
        print(tag, "无 eval_auto 文件"); continue
    best = (0.0, None)
    pts = []
    for f in fs:
        try:
            d = json.load(open(f))
            s = d.get("ssim_mean") or d.get("ssim")
            st = int(f.split("eval_auto_")[-1].split(".")[0])
        except Exception:
            continue
        if s:
            pts.append((st, s))
            if s > best[0]:
                best = (s, st)
    pts.sort()
    tail = pts[-8:] if len(pts) > 8 else pts
    print("%s: best %.4f @ step %s (点数 %d, 首 %d 尾 %d)" % (tag, best[0], best[1], len(pts), pts[0][0], pts[-1][0]))
    for st, s in tail:
        print("   %6d  %.4f" % (st, s))