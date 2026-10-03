# -*- coding: utf-8 -*-
"""_loss_curve.py — 对比当前线与原版 4ch 线的 Diff 轨迹, 判断是否平台."""
import glob
import os
import re
import sys

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAT = re.compile(r"step=(\d+)\) Diff: ([0-9.]+)")
LRP = re.compile(r"LR: ([0-9.e+-]+)")


def curve(paths):
    pts = []
    for p in paths:
        for line in open(p, encoding="utf-8", errors="replace"):
            m = PAT.search(line)
            if m:
                lr = LRP.search(line)
                pts.append((int(m.group(1)), float(m.group(2)),
                            float(lr.group(1)) if lr else None))
    pts.sort()
    return pts


def show(tag, paths, grid):
    pts = curve(paths)
    if not pts:
        print(f"\n[{tag}] 无数据 ({paths})")
        return
    print(f"\n[{tag}] 共 {len(pts)} 条; 步数范围 {pts[0][0]}..{pts[-1][0]}")
    print(f"  {'step':>8s} {'Diff':>8s} {'LR':>10s}")
    last = -1
    for s, d, lr in pts:
        if s >= last + grid:
            print(f"  {s:8d} {d:8.4f} {('%.3e' % lr) if lr else '-':>10s}")
            last = s


show("当前 v11_pretrain_Sp2_base_sym (4ch, 158K)",
     ["logs/v11_pretrain_Sp2_base_sym/train_r3.log"], 1500)
show("原版 4ch v11_pretrain_Sp2_base (54K)",
     sorted(glob.glob("logs/v11_pretrain_Sp2_base/*.log")), 1500)

# 最近 2000 步的波动
pts = curve(["logs/v11_pretrain_Sp2_base_sym/train_r3.log"])
if len(pts) > 40:
    tail = [d for _, d, _ in pts[-40:]]
    print(f"\n[最近 40 条] min={min(tail):.4f} max={max(tail):.4f} "
          f"均值={sum(tail)/len(tail):.4f} 波动={max(tail)-min(tail):.4f}")
