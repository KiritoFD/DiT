# -*- coding: utf-8 -*-
"""build_midstrict_split.py — 建第 4 份固定切分 CSV: **mid-strict**。

定义 (用户 2026-10-07 裁定): mid-strict = **mid 档 ∩ 组合未见**。
   mid 档         : 三份固定 CSV 里的 eval200_split_mid.csv (v68 每样本 SSIM 降序 63..125, 63 张)
   组合未见        : exp-std/csv/eval200_combo_unseen.csv (112 张) —— 即 (书家,书体,字)
                    三元组**未在训练清单 train.csv 里出现过**的样本 (审计见 tools/audit_eval_leakage*.py)

为什么要这一档: 现有 mid 档里 25/63 (40%) 的样本组合在训练里出现过, 且该子集对**所有模型**
都更容易 (全 187 张上 v68 +0.0576 / v66 +0.0771) -> 会抬高绝对值。mid-strict 把这部分剔掉,
给出"日常中间难度 + 无组合记忆"的口径。

产物: exp-std/csv/eval200_split_midstrict.csv  + 打印三重交叉核对
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MID = "exp-std/csv/eval200_split_mid.csv"
UNSEEN = "exp-std/csv/eval200_combo_unseen.csv"
TRAIN = "exp-std/csv/train.csv"
OUT = "exp-std/csv/eval200_split_midstrict.csv"
K3 = ("calligrapher", "script", "character")


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    mid = rows(MID)
    unseen = rows(UNSEEN)
    un_key = {r["image_path"] for r in unseen}
    out = [r for r in mid if r["image_path"] in un_key]

    # 独立核对 1: 直接用 train.csv 的组合集合重算"未见", 应与 combo_unseen 一致
    seen3 = {tuple(str(r.get(k, "")).strip() for k in K3) for r in rows(TRAIN)}
    recalc_unseen = {r["image_path"] for r in rows("exp-std/csv/eval200_fixed.csv")
                     if tuple(str(r.get(k, "")).strip() for k in K3) not in seen3}
    same = un_key == recalc_unseen
    print(f"  [核对1] combo_unseen({len(un_key)}) == 用 train.csv 重算的未见集({len(recalc_unseen)}) : "
          f"{'✓' if same else '✗ 不一致!'}")

    # 独立核对 2: 输出里不应有任何"组合见过"的样本
    bad = [r for r in out if tuple(str(r.get(k, "")).strip() for k in K3) in seen3]
    print(f"  [核对2] 输出中组合仍见过的样本: {len(bad)}  {'✓' if not bad else '✗'}")

    # 独立核对 3: mid 档内部被剔掉多少
    print(f"  [核对3] mid 档 {len(mid)} 张 -> mid-strict {len(out)} 张 "
          f"(剔掉 {len(mid) - len(out)} 张组合见过的, 占 {100*(len(mid)-len(out))/max(1,len(mid)):.0f}%)")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mid[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"  ✓ 已写 {OUT} ({len(out)} 行)")


if __name__ == "__main__":
    main()
