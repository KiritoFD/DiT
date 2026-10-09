# -*- coding: utf-8 -*-
"""audit_eval_dirs.py — 审计根目录 eval/ 下每个模型目录的完整性与命名规则。

回答三个问题:
  1. 每个目录覆盖了 eval200fix 的哪 187 个索引? 缺哪几个?
  2. 有没有目录里混着两套图 (如 moyi_12ch 有 374 张 = 2x187)?
  3. 文件名前缀是什么 (会不会与 gt 对不上)?

索引以 exp-std/csv/eval200_fixed.csv 的行为准 (第 i 行 <-> idx i),
因为我们的 eval_samples 与 per-sample CSV 都是按 idx 对齐的。
"""
import os
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = "eval"
N = 187


def nums(files):
    """从文件名里抠出所有整数 -> 集合"""
    out = set()
    for f in files:
        for m in re.findall(r"\d+", os.path.splitext(f)[0]):
            out.add(int(m))
    return out


def main():
    dirs = sorted(d for d in os.listdir(EVAL_DIR)
                  if os.path.isdir(os.path.join(EVAL_DIR, d)))
    print(f"根目录 {EVAL_DIR}/ 下 {len(dirs)} 个目录\n")
    print(f"  {'目录':<14} {'文件数':>6} {'覆盖 idx 数':>10} {'多余的 idx(>186)':>16}  样例文件名")
    print("  " + "-" * 104)
    detail = {}
    for d in dirs:
        p = os.path.join(EVAL_DIR, d)
        files = sorted(os.listdir(p))
        idx = nums(files)
        in_range = {i for i in idx if 0 <= i < N}
        over = sorted(i for i in idx if i >= N)
        detail[d] = (files, in_range)
        sample = ", ".join(files[:3])
        print(f"  {d:<14} {len(files):>6} {len(in_range):>10} "
              f"{str(over[:6]) + ('...' if len(over) > 6 else ''):>16}  {sample}")

    print("\n=== 缺失索引 (相对 0..186) ===")
    full = set(range(N))
    for d, (files, in_range) in detail.items():
        miss = sorted(full - in_range)
        status = "完整 187" if not miss else f"缺 {len(miss)} 个: {miss[:12]}{'...' if len(miss) > 12 else ''}"
        print(f"  {d:<14} {status}")

    print("\n=== 每个索引被多少目录覆盖 (用来看能不能做全模型同集对照) ===")
    cov = defaultdict(list)
    for d, (files, in_range) in detail.items():
        for i in in_range:
            cov[i].append(d)
    n_models = len(dirs)
    by_count = defaultdict(int)
    for i in full:
        by_count[len(cov[i])] += 1
    for k in sorted(by_count, reverse=True):
        print(f"  被 {k:2d}/{n_models} 个目录覆盖的索引: {by_count[k]:3d} 个")

    # 全模型共有的索引 = 可以严格同集对照的最大集合
    common = sorted(i for i in full if len(cov[i]) == n_models)
    print(f"\n  → 全部 {n_models} 个目录都有的索引: {len(common)} 个"
          f"{'  (可做严格同集对照)' if len(common) == N else '  (不足 187, 只能在该子集上同集对照)'}")


if __name__ == "__main__":
    main()
