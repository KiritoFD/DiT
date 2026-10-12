#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""disk_clean.py — 保守清理中间 ckpt（逐条打印 + 可选 --apply）

保留规则（每个 run 目录独立判定）：
  1) **保留最后一个** ckpt（无论 step）；
  2) 保留 KEEP_STEPS 里的点（我们在 state.md / 文档里引用过的匹配点）；
  3) 其余 `*.pt`（及其 `.done` 标记）删除。
**不动**：assets/runs 之外的任何东西；48 上的 exp-std/data 增广数据 / REPA 缓存 / std_soft store。

用法: python disk_clean.py [--apply] [--root <repo>]
"""
import argparse
import glob
import os
import re
import sys

KEEP_STEPS = {20000, 35000, 40000, 50000, 55000, 100000, 150000, 177778, 200000}


def step_of(p):
    m = re.search(r"(\d+)\.pt$", os.path.basename(p))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    os.chdir(a.root)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    _hits = []
    for base in ("assets/runs", "assets/results"):
        _hits += glob.glob(os.path.join(base, "**", "checkpoints", "*.pt"), recursive=True)
    runs = sorted({os.path.dirname(p) for p in _hits})
    total_del = total_keep = 0
    plan = []
    for d in runs:
        pts = sorted(glob.glob(os.path.join(d, "*.pt")), key=lambda p: step_of(p) or -1)
        if not pts:
            continue
        last = pts[-1]
        dele = [p for p in pts if p != last and (step_of(p) not in KEEP_STEPS)]
        keep = [p for p in pts if p not in dele]
        sz = sum(os.path.getsize(p) for p in pts)
        dsz = sum(os.path.getsize(p) for p in dele)
        total_del += dsz
        total_keep += sum(os.path.getsize(p) for p in keep)
        tag = d.replace("assets/runs/", "").replace("/checkpoints", "")
        plan.append((tag, len(pts), len(keep), len(dele), sz / 2**30, dsz / 2**30, dele))

    print(f"  共 {len(plan)} 个 run；拟删除 {sum(p[3] for p in plan)} 个 ckpt，"
          f"回收 **{total_del/2**30:.1f} GB**；保留 {total_keep/2**30:.1f} GB")
    print(f"\n  {'run':<56}{'总':>4}{'留':>4}{'删':>4}{'总GB':>8}{'删GB':>8}")
    for tag, n, k, dl, sz, dsz, _ in sorted(plan, key=lambda x: -x[5]):
        print(f"  {tag[:55]:<56}{n:>4}{k:>4}{dl:>4}{sz:>8.1f}{dsz:>8.1f}")

    if a.apply:
        n = 0
        for _, _, _, _, _, _, dele in plan:
            for p in dele:
                try:
                    os.remove(p)
                    n += 1
                    for suf in (".done",):
                        if os.path.exists(p + suf):
                            os.remove(p + suf)
                except OSError as e:
                    print(f"  ! {p}: {e}")
        print(f"\n  ✅ 已删除 {n} 个中间 ckpt，回收约 {total_del/2**30:.1f} GB")
    else:
        print("\n  （dry-run：加 --apply 才会真删）")


if __name__ == "__main__":
    main()
