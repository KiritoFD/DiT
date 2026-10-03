#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_paired.py — 两个臂的**逐样本配对**比较（同一批字、同一份噪声 seed=0）。

为什么不能只看两个 ssim 均值相减: n=100 时独立样本 SE≈0.02，差 0.01 什么都说明不了。
但 in_mem_eval 的噪声是固定 seed、GT 是同一批字 -> 两臂是**配对**观测，
配对差的标准差远小于独立情形，同样的差值可以显著得多。

用法:
    python _sync_work/fs6_paired.py <A 的 eval_stdskel_batch.csv> <B 的 ...> [step]
    python _sync_work/fs6_paired.py --auto 沈周-行        # 自动找 base_mean/base_row_pt
"""
import csv
import glob
import os
import statistics as st
import sys

os.chdir("/root/Workspace/xy/DiT")


def find_csv(results_dir):
    """结果目录下可能有多个时间戳 run（重跑过），取最新的 eval_stdskel_batch.csv。"""
    cands = glob.glob(os.path.join(results_dir, "**", "eval_stdskel_batch.csv"),
                      recursive=True)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def load(path, step=None, name="fewshot"):
    out = {}
    if not path or not os.path.isfile(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("set") != name:
            continue
        if step is not None and int(float(r["step"])) != step:
            continue
        # 落盘的 img_id 列在 few-shot 集里其实是图路径 -> 用"字"当配对键（同主题内唯一）
        key = r.get("char") or r.get("img_id") or r.get("idx")
        out[key] = (float(r["ssim"]), r.get("char", ""))
    return out


def compare(lab_a, a, lab_b, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        print(f"  无可配对样本（{len(a)} vs {len(b)}）")
        return
    d = [b[k][0] - a[k][0] for k in keys]
    n = len(d)
    mean = st.fmean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    se = sd / (n ** 0.5) if sd else 0.0
    wins = sum(1 for x in d if x > 0)
    t = mean / se if se else float("nan")
    # 双侧符号检验（正态近似）
    z = (wins - n / 2) / (n ** 0.5 / 2) if n else float("nan")
    print(f"  {lab_a} -> {lab_b}:  n={n}  Δssim 均值={mean:+.4f}  "
          f"配对SE={se:.4f}  t={t:+.2f}  |  {lab_b} 更好 {wins}/{n} 张"
          f"  符号检验 z={z:+.2f}({'显著' if abs(z) > 1.96 else '不显著'})")
    big = sorted(zip(keys, d), key=lambda z2: -abs(z2[1]))[:5]
    print("    变化最大: " + ", ".join(f"{a[k][1]}{v:+.3f}" for k, v in big))


def _dir(d):
    return load(find_csv(d))


if __name__ == "__main__":
    if sys.argv[1] == "--auto":
        for topic in sys.argv[2:]:
            print(f"\n[{topic}] 不训练 baseline: mean_scaled vs dino(row_pt) —— 只差"
                  f"新行怎么写，零梯度")
            compare("mean", _dir(f"/tmp/_fs6_{topic}_base_mean_scaled"),
                    "dino", _dir(f"/tmp/_fs6_{topic}_base_row_pt"))
    else:
        print(f"{sys.argv[1]}  vs  {sys.argv[2]}")
        compare("A", _dir(sys.argv[1]), "B", _dir(sys.argv[2]))
