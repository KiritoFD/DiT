# -*- coding: utf-8 -*-
"""列出当前 XL（v3a_xl_highdim）实验的 eval_samples step 目录及样本数。"""
import os, glob

dirs = sorted(glob.glob("5script/results/v3a_xl_highdim/*/checkpoints/eval_samples/step*"))
print(f"XL eval_samples step 目录 ({len(dirs)}):")
for d in dirs:
    samples = len([f for f in os.listdir(d) if f.startswith("sample")])
    print(f"  {os.path.basename(d)}: {samples} samples")
