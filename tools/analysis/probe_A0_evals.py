#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_A0_evals.py — A0 的 per-step 评测图结构"""
import glob
import os
import sys

import numpy as np
from PIL import Image

os.chdir("/home/ds/Workspace/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = glob.glob("assets/results/v83_v68_noREPA/*/eval_samples_ctrl")
print(f"  找到 {len(D)} 个 eval_samples_ctrl: {D}")
if not D:
    print("  ✗ A0 没存 per-step 评测图 -> 需要另行采样生成")
    sys.exit(0)
D = D[0]
steps = sorted(os.listdir(D))
print(f"  step 目录数={len(steps)}: {' '.join(s.replace('step','') for s in steps)}")
for s in steps[:2] + steps[-2:]:
    p = os.path.join(D, s)
    pngs = glob.glob(os.path.join(p, "**", "*.png"), recursive=True)
    print(f"\n  {s}: {len(pngs)} png")
    g = sorted(glob.glob(os.path.join(p, "eval200fix", "g[0-9]*.png")))
    t = sorted(glob.glob(os.path.join(p, "eval200fix", "gt[0-9]*.png")))
    print(f"     eval200fix/g*.png={len(g)}  gt*.png={len(t)}")
    for q in g[:4]:
        a = np.asarray(Image.open(q).convert("L"))
        print(f"     {os.path.relpath(q, p):<30} 墨量={float((a<128).mean()):.4f}")
