#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize a run's own eval_auto trend (mean/min) to compare against strict eval."""
import sys, glob, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = sys.argv[1]
def f(x, fmt="{:.4f}"):
    return fmt.format(x) if x is not None else "-"
for fn in sorted(glob.glob(os.path.join(root, "checkpoints", "eval_auto_*.json"))):
    d = json.load(open(fn))
    step = d.get("step", os.path.basename(fn))
    print(f"step {step}: ssim={f(d.get('ssim'))} ssim_min={f(d.get('ssim_min'))} "
          f"mse={f(d.get('mse'))} skel={f(d.get('skel_iou'))} n={d.get('n')}")
