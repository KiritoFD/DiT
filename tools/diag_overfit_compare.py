# -*- coding: utf-8 -*-
"""diag_overfit_compare.py — 横向对比各 run 的 seen/strict 曲线, 判断"gap 扩大"是否异常.

只读。扫描 assets/results/*/eval_stdskel_summary.csv, 输出:
  run / 最后 step / seen / strict / gap, 以及各 run 的 seen-strict gap 随 step 的变化率。
用于回答: 本轮 gap +0.20 且仍在扩大, 是"这个项目的常态"还是"本轮独有的过拟合"。
"""
import csv
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = []
for p in sorted(glob.glob("assets/results/*/eval_stdskel_summary.csv")):
    exp = os.path.basename(os.path.dirname(p))
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            d.setdefault(int(r["step"]), {})[r["set"]] = float(r["ssim_mean"])
        except Exception:
            pass
    st = sorted(s for s in d if "seen" in d[s] and "strict" in d[s])
    if not st:
        continue
    last = st[-1]
    # 取最大 strict 及其对应 step
    best_s = max(st, key=lambda s: d[s]["strict"])
    # 末段斜率 (最后 4 个点)
    tail = st[-4:] if len(st) >= 4 else st
    ds = tail[-1] - tail[0]
    dseen = (d[tail[-1]]["seen"] - d[tail[0]]["seen"]) / ds * 10000 if ds else 0.0
    dstrict = (d[tail[-1]]["strict"] - d[tail[0]]["strict"]) / ds * 10000 if ds else 0.0
    rows.append((exp, last, d[last]["seen"], d[last]["strict"], d[last]["seen"] - d[last]["strict"],
                 best_s, d[best_s]["strict"], dseen, dstrict))

rows.sort(key=lambda r: -r[6])
print(f"{'run':52s} {'lastStep':>9} {'seen':>7} {'strict':>7} {'gap':>7} "
      f"{'bestStrict':>10} {'@step':>8} {'dSeen/10k':>9} {'dStrict/10k':>11}")
print("-" * 140)
for r in rows:
    print(f"{r[0][:52]:52s} {r[1]:9d} {r[2]:7.4f} {r[3]:7.4f} {r[4]:+7.4f} "
          f"{r[6]:10.4f} {r[5]:8d} {r[7]:+9.4f} {r[8]:+11.4f}")
print(f"\n共 {len(rows)} 个 run 有 eval 记录")
