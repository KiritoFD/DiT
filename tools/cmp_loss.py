# -*- coding: utf-8 -*-
"""cmp_loss.py — 对比多个 run 的 step->loss / LR 曲线 (含同配方历史 run)。"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAT = re.compile(r"step=0*(\d+)\)\s*Diff:\s*([0-9.]+).*?LR:\s*([0-9.eE+-]+)")

RUNS = {
    "clean(本次)": "logs/v11_pretrain_Sp2_base_sym_clean/train.log",
    "sym2(旧数据同配方)": "logs/v11_pretrain_Sp2_base_sym2/train.log",
    "sym(旧数据)": "logs/v11_pretrain_Sp2_base_sym/train.log",
    "wz(12ch)": "logs/v11_pretrain_Sp2_base_wz/train.log",
}


def series(path, mx=200000):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        m = PAT.search(line)
        if m:
            s = int(m.group(1))
            out[s] = (float(m.group(2)), m.group(3))
    return out


print(f"{'run':24s} " + " ".join(f"{s:>10d}" for s in
                                 (50, 100, 200, 400, 800, 1600, 3000, 5000, 10000)))
for name, p in RUNS.items():
    d = series(p)
    if not d:
        print(f"{name:24s} (无日志 {p})")
        continue
    row = []
    for s in (50, 100, 200, 400, 800, 1600, 3000, 5000, 10000):
        v = d.get(s)
        row.append(f"{v[0]:10.4f}" if v else f"{'-':>10s}")
    print(f"{name:24s} " + " ".join(row))

print()
for name, p in RUNS.items():
    d = series(p)
    if not d:
        continue
    smax = max(d)
    lr = d[smax][1]
    print(f"{name:24s} 最新 step={smax:7d}  Diff={d[smax][0]:.4f}  LR={lr}")

# 当前 run 的 LR 是否还在 warmup
cur = series(RUNS["clean(本次)"])
if cur:
    mx = max(cur)
    print(f"\n当前 run: max_step={mx}, LR={cur[mx][1]}")
