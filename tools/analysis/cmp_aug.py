# -*- coding: utf-8 -*-
"""cmp_aug.py — A1(morphE, REPA=0) vs 已有消融臂(sym3x_noOT / noaug_noOT / ...) 匹配步数对照

⚠ 混杂声明: A1 是 REPA=0, arm2/arm4 是 REPA=0.03 -> 混入实测 1~7% 的小偏差 ✗
   (A0 实验已量化: ssim/lpips 无差异, ink_iou/hole/nn_ssim 差 1~7%)
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIGHER = {"ssim", "ink_iou", "ink_ssim", "skel_iou", "tgt_spec", "nn_mean", "nn_ssim"}
SHOW = ["ssim", "lpips", "frag", "ink_iou", "ink_ssim", "hole", "nn_ssim", "skel_iou"]
NAMES = ["A1_morphE", "arm2_sym3x", "arm4_noaug"]


def load(p):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            st = int(r["step"])
        except Exception:
            continue
        d.setdefault(r["name"], {})[st] = {
            k: float(v) for k, v in r.items()
            if k not in ("name", "step", "n") and v not in ("", None)}
    return d


D = load("_local_ckpt/eval_aug_full.csv")
for nm in NAMES:
    if nm in D:
        print(f"  {nm:<16} 评测点 {min(D[nm])}..{max(D[nm])}  共 {len(D[nm])} 个")
com = sorted(set.intersection(*[set(D[n]) for n in NAMES if n in D]))
print(f"\n  三者共同评测点: {com}\n")

for m in SHOW:
    print(f"  ── {m} ({'越高越好' if m in HIGHER else '越低越好'})")
    print("     " + f"{'step':>7}" + "".join(f"{n.split('_')[0]:>12}" for n in NAMES if n in D)
          + "   A1 vs arm2   A1 vs arm4")
    for s in com:
        row = f"     {s:>7}"
        for n in NAMES:
            v = D[n].get(s, {}).get(m)
            row += f"{v:>12.4f}" if v is not None else f"{'—':>12}"
        a2 = D["arm2_sym3x"].get(s, {}).get(m)
        a4 = D["arm4_noaug"].get(s, {}).get(m)
        a1 = D["A1_morphE"].get(s, {}).get(m)
        def dd(x, y):
            if x is None or y is None:
                return "     —"
            d = x - y
            good = (d > 0) if m in HIGHER else (d < 0)
            return f"{d:+.4f}{'+' if good else '-'}"
        print(row + f"   {dd(a1, a2):>10}   {dd(a1, a4):>10}")
    print()

print("  === 平均（三者共同点）===")
for m in SHOW:
    line = f"  {m:<10}"
    for n in NAMES:
        vs = [D[n][s][m] for s in com if m in D[n].get(s, {})]
        line += f"  {n.split('_')[0]}={np.mean(vs):.4f}" if vs else f"  {n}=—"
    print(line)
print("""
  读法: 每格后的 +/- 已按指标方向标注 ✓（+ = A1 更好）
        A1 是 REPA=0, arm2/arm4 是 REPA=0.03 -> 混杂了小偏差(1~7%) ✗, 但 aug 族之间的差异若达 0.005+ 则远超该噪声 ✓
""")
