# -*- coding: utf-8 -*-
"""列出 eval_samples 各 step 目录的 png 数量。"""
import os, glob

base = "5script/results/v3a/*/checkpoints/eval_samples"
for d in sorted(glob.glob(base + "/step*")):
    pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    print(os.path.basename(d), len(pngs), "png ->", pngs)
