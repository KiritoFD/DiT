# -*- coding: utf-8 -*-
"""读取全量清洗报告，输出修复统计与质量评估。"""
import os, sys, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")
import numpy as np

for name, path in [("TRAIN", "assets/clean_report_train_B.csv"),
                   ("EVAL", "assets/clean_report_eval_B.csv")]:
    if not os.path.isfile(path):
        print(f"{name}: missing")
        continue
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    N = len(rows)
    print(f"\n{'='*60}")
    print(f"{name}: {N} 张 (scheme B)")
    print(f"{'='*60}")

    def col(k):
        return np.array([float(r[k]) for r in rows], dtype=float)

    ch = col("changed"); inv = col("inverted"); cr = col("cropped")
    ic = col("ink_change"); mk = col("main_keep")
    ncc_b = col("n_cc_before"); ncc_a = col("n_cc_after")
    sl = col("small_cc_left")

    print(f"实际修复 changed=1:  {int(ch.sum())} ({ch.mean()*100:.2f}%)")
    print(f"  其中 反相反转:      {int(inv.sum())} ({inv.mean()*100:.2f}%)")
    print(f"  其中 黑边crop:      {int(cr.sum())} ({cr.mean()*100:.2f}%)")
    print(f"\n质量:")
    print(f"  主连通域保留 main_keep: mean={mk.mean():.5f} median={np.median(mk):.5f}")
    print(f"     main_keep<0.999 (笔画被删): {(mk<0.999).sum()} ({(mk<0.999).mean()*100:.2f}%)")
    print(f"     main_keep<0.95  (严重误删): {(mk<0.95).sum()} ({(mk<0.95).mean()*100:.2f}%)")
    print(f"  墨量变化 ink_change:   mean={ic.mean():.5f} median={np.median(ic):.5f}")
    print(f"     |ink_change|>0.05:  {(np.abs(ic)>0.05).sum()} ({(np.abs(ic)>0.05).mean()*100:.2f}%)")
    print(f"     |ink_change|>0.10:  {(np.abs(ic)>0.10).sum()} ({(np.abs(ic)>0.10).mean()*100:.2f}%)")
    print(f"  连通域数: before mean={ncc_b.mean():.2f} -> after mean={ncc_a.mean():.2f}")
    print(f"  小噪点残留 small_cc_left: mean={sl.mean():.4f} (修复前见 scan)")
    print(f"     small_cc_left>0:   {(sl>0).sum()} ({(sl>0).mean()*100:.2f}%)")
