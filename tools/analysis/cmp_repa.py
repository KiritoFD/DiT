# -*- coding: utf-8 -*-
"""cmp_repa.py — A0(REPA=0) vs v68(REPA=0.03) 全指标逐点对照

判据（用户）: 关 REPA 是否仍有负面表现 —— 不只看 SSIM，要全指标
符号约定: ssim/ink_iou/ink_ssim/skel_iou/nn_ssim/tgt_spec 越高越好;
          lpips/mse/frag/hole 越低越好
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIGHER = {"ssim", "ink_iou", "ink_ssim", "skel_iou", "tgt_spec", "nn_mean", "nn_ssim", "cal_enrich"}
SHOW = ["ssim", "lpips", "frag", "ink_iou", "ink_ssim", "skel_iou", "hole", "nn_ssim", "tgt_spec", "mse"]


def load(p):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            st = int(r["step"])
        except Exception:
            continue
        d[st] = {k: (float(v) if v not in ("", None) else None) for k, v in r.items()
                 if k not in ("name", "step") and v not in ("", None)}
    return d


V68 = load("_local_ckpt/eval_v68_full.csv")
A0 = load("_local_ckpt/eval_A0_full.csv")
com = sorted(set(V68) & set(A0))
print(f"  匹配评测点: {com}")
print(f"  (v68 共 {len(V68)} 点 / A0 共 {len(A0)} 点)\n")

print("  === 逐点：Δ = A0 − v68（绿=更好, 红=更差, 按各指标方向）===")
hdr = "  " + f"{'step':>6} |"
for m in SHOW:
    hdr += f" {m:>9}"
print(hdr)
for s in com:
    line = f"  {s:>6} |"
    for m in SHOW:
        a, b = A0[s].get(m), V68[s].get(m)
        if a is None or b is None:
            line += f" {'—':>9}"
            continue
        d = a - b
        good = (d > 0) if m in HIGHER else (d < 0)
        mark = "+" if good else "-"
        line += f" {d:>+8.4f}{mark}"
    print(line)

print("\n  === 平均 Δ（A0 − v68）与胜率 ===")
print(f"  {'指标':<10} {'平均Δ':>10} {'A0更好点数':>10} {'方向':>6}   判定")
for m in SHOW:
    ds = []
    for s in com:
        a, b = A0[s].get(m), V68[s].get(m)
        if a is None or b is None:
            continue
        ds.append(a - b)
    if not ds:
        continue
    ds = np.array(ds)
    wins = int((ds > 0).sum() if m in HIGHER else (ds < 0).sum())
    worse = int((ds < 0).sum() if m in HIGHER else (ds > 0).sum())
    verdict = ("A0 全面更好 ✓" if worse == 0 and wins > 0 else
               "A0 全面更差 ✗" if wins == 0 and worse > 0 else
               f"互有胜负（A0 好 {wins} / 差 {worse}）")
    print(f"  {m:<10} {ds.mean():>+10.5f} {wins:>6}/{len(ds):<3} "
          f"{'↑好' if m in HIGHER else '↓好':>6}   {verdict}")

print(f"""
  读法: 每格 Δ 后的 +/- 已按"该指标是否更好"标注 ✓
        SSIM 单独看会被稀疏 GT 的白图基线污染（见 white_baseline.py），
        所以必须同时看 ink_iou（对空白敏感 ✓）与 frag/hole（结构/空洞）✓
""")
