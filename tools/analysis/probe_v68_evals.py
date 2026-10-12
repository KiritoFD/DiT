#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_v68_evals.py — v68 的 per-step 评测图结构（能否直接用于匹配步数视觉对照）"""
import glob
import os
import sys

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = glob.glob("assets/runs/v68_family/v68_aug_sp_c2ot/*/eval_samples_ctrl")
print(f"  找到 {len(D)} 个 eval_samples_ctrl")
if not D:
    sys.exit(0)
D = D[0]
steps = sorted(os.listdir(D))
print(f"  step 目录数={len(steps)}  范围: {steps[0]} .. {steps[-1]}")
print(f"  全部: {' '.join(s.replace('step','') for s in steps)}")

for s in steps[:3] + steps[-3:]:
    p = os.path.join(D, s)
    pngs = glob.glob(os.path.join(p, "**", "*.png"), recursive=True)
    subdirs = sorted({os.path.relpath(os.path.dirname(x), p) for x in pngs})
    print(f"\n  {s}: {len(pngs)} png;  子目录={subdirs[:4]}")
    if pngs:
        names = sorted(glob.glob(os.path.join(p, "**", "g[0-9]*.png"), recursive=True)) or \
                sorted(glob.glob(os.path.join(p, "**", "*.png"), recursive=True))
        for q in names[:4]:
            a = np.asarray(Image.open(q).convert("L"))
            print(f"     {os.path.relpath(q, p):<34} {a.shape} 墨量={float((a<128).mean()):.4f}")
