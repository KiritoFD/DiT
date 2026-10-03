#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_report.py — 把 logs/v15_series/v15_fs6/*.log 汇成一张表（baseline / 训练轨迹 / Diff）。

用法: /opt/conda/envs/cu121/bin/python _sync_work/fs6_report.py [日志目录]
"""
import glob
import os
import re
import sys
import time
from collections import defaultdict

LOGD = sys.argv[1] if len(sys.argv) > 1 else "/root/Workspace/xy/DiT/logs/v15_series/v15_fs6"
RE_EVAL = re.compile(r"set=(\w+) n=(\d+) ssim=([\d.]+) \(med=([\d.]+)\) "
                     r"mse=([\d.]+) lpips=([\d.]+)")
RE_DIFF = re.compile(r"\(step=(\d+)\).*?Diff: ([\d.]+)")

series = defaultdict(list)
for path in sorted(glob.glob(os.path.join(LOGD, "*.log"))):
    key = os.path.basename(path)
    key = re.sub(r"_\d{4}-\d{6}(?=\.log$)", "", key)
    ev, df = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RE_EVAL.search(line)
        if m:
            ev.append((int(re.search(r"step=(\d+)", line).group(1)),
                       m.group(1), float(m.group(4)), float(m.group(3)),
                       float(m.group(6))))
        m2 = RE_DIFF.search(line)
        if m2:
            df.append((int(m2.group(1)), float(m2.group(2))))
    series[key] = (ev, df)

if not series:
    print(f"没有日志: {LOGD}/*.log")
    sys.exit(0)

print(f"{'log':<46}{'n_eval':>7}{'ssim 首':>9}{'ssim 峰':>9}{'ssim 末':>9}"
      f"{'med峰':>8}{'lpips峰':>9}  Diff 首→末")
for key, (ev, df) in sorted(series.items()):
    if not ev:
        age = (time.time() - os.path.getmtime(os.path.join(LOGD, key + ".log"))) \
            if os.path.exists(os.path.join(LOGD, key + ".log")) else -1
        tail = "" if age < 180 else "   —— 3 分钟没动了，可能挂了"
        print(f"{key:<46}{0:>7}   还没出 eval{tail}")
        continue
    first, best, last = ev[0], max(ev, key=lambda z: z[3]), ev[-1]
    dif = f"{df[0][1]:.3f}→{df[-1][1]:.3f} ({df[-1][0]-df[0][0]:+d}步)" if df else "—"
    print(f"{key:<46}{len(ev):>7}{first[3]:>9.4f}{best[3]:>9.4f}{last[3]:>9.4f}"
          f"{best[2]:>8.4f}{best[4]:>9.4f}  {dif}")
    if len(ev) > 2:
        traj = " ".join(f"{s - ev[0][0]}:{v:.4f}" for s, _, _, v, _ in ev)
        print(f"{'':<6}轨迹(stepΔ:ssim) {traj}")

print("\n参照系: 已训书家/书体在 seen 上 ssim≈0.57, Diff 0.24-0.31; "
      "few-shot 的判据是 **同主题同口径下** 训练/不同 init 相对 baseline 的提升。")
