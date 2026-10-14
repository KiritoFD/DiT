# -*- coding: utf-8 -*-
"""cmp_aug100k.py — 增广族四臂(A1..A4) vs 已有消融四臂 的 100k 终值完整对照

口径: 全部 batch 512 x 100k 步 (= 51.2M 样本, 同样本 ✓), 同一 187 张评测集
      ⚠ A1..A4 是 REPA=0; 已有四臂是 REPA=0.03 -> 混入实测 1~7% 的小偏差
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MS = ["ssim", "lpips", "frag", "ink_iou", "ink_ssim", "hole", "nn_ssim", "skel_iou", "tgt_spec"]
HIGH = {"ssim", "ink_iou", "ink_ssim", "skel_iou", "nn_ssim", "tgt_spec"}
NEW = ["A1_morphE", "A2_elastic", "A3_grid", "A4_affine"]
OLD = ["arm4_noaug", "arm2_sym3x", "arm1_noaug_c2ot", "arm3_sym3x_naiveOT"]
LABEL = {"A1_morphE": "A1 morphE(形态学)", "A2_elastic": "A2 elastic(弹性形变)",
         "A3_grid": "A3 grid(网格畸变)", "A4_affine": "A4 affine(仿射)",
         "arm4_noaug": "arm4 noaug(无增强)", "arm2_sym3x": "arm2 sym3x(原增强)",
         "arm1_noaug_c2ot": "arm1 noaug+C2OT", "arm3_sym3x_naiveOT": "arm3 sym3x+naiveOT"}


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
STEPS = [100000, 95000, 90000, 85000, 80000, 75000]

for name in NEW + OLD:
    got = sorted(D.get(name, {}))
    print(f"  {name:<22} 点数={len(got):>3}  范围 {got[0] if got else '-'}..{got[-1] if got else '-'}")

print("\n" + "=" * 118)
print("  【100k 终值】 全部 batch512 x 100k = 51.2M 样本, 同 187 张评测集")
print("=" * 118)
print(f"  {'臂':<24}" + "".join(f"{m:>10}" for m in MS))
print("  " + "-" * 114)
for name in NEW + OLD:
    v = D.get(name, {}).get(100000)
    if not v:
        print(f"  {LABEL.get(name, name):<24}" + "".join(f"{'—':>10}" for _ in MS))
        continue
    print(f"  {LABEL.get(name, name):<24}" + "".join(
        f"{v[m]:>10.4f}" if m in v else f"{'—':>10}" for m in MS))

print("\n  【各指标排名 (100k, 共 8 臂)】")
for m in MS:
    vals = []
    for n in NEW + OLD:
        v = D.get(n, {}).get(100000, {}).get(m)
        if v is not None:
            vals.append((n, v))
    if not vals:
        continue
    vals.sort(key=lambda x: x[1], reverse=(m in HIGH))
    rank = "  ".join(f"{i+1}.{LABEL[n].split('(')[0].split()[0]}={v:.4f}" for i, (n, v) in enumerate(vals[:4]))
    print(f"    {m:<9} {rank}")

print("\n  【增广族四臂 vs 无增强(arm4) 的 Δ】 + = 增强臂更好")
print(f"  {'臂':<24}" + "".join(f"{m:>10}" for m in MS))
base = D.get("arm4_noaug", {}).get(100000, {})
for name in NEW:
    v = D.get(name, {}).get(100000)
    if not v:
        continue
    row = f"  {LABEL.get(name, name):<24}"
    for m in MS:
        if m not in v or m not in base:
            row += f"{'—':>10}"
            continue
        d = v[m] - base[m]
        good = (d > 0) if m in HIGH else (d < 0)
        row += f"{d:>+9.4f}{'+' if good else '-'}"
    print(row)

print("\n  【增广族四臂互比: 与最好者的差】")
print("\n  【配对检验: 每个增广臂 vs arm4(noaug) 在全部共同评测点上的胜负 (100k 以下)】")
allpts = sorted(set.intersection(*[set(D.get(n, {})) for n in NEW + OLD]) or set())
if allpts:
    print(f"  共同评测点 {len(allpts)} 个: {allpts}")
    # 只在四臂都有的点比较
    for m in MS:
        print(f"    {m:<9}", end="")
        for name in NEW:
            w = 0
            tot = 0
            for s in allpts:
                a = D.get(name, {}).get(s, {}).get(m)
                b = base.get(m) if s == 100000 else D.get("arm4_noaug", {}).get(s, {}).get(m)
                if a is None or b is None:
                    continue
                tot += 1
                d = a - b
                if (d > 0) if m in HIGH else (d < 0):
                    w += 1
            print(f"   {LABEL[name].split('(')[0].split()[0]} {w}/{tot}", end="")
        print()