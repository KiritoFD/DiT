#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, json

for pat in [
    "/root/Workspace/xy/DiT/5script/results/s6_top6_diffonly/*resume*/checkpoints/eval_auto_*.json",
    "/root/Workspace/xy/DiT/5script/results/s6_top6_*/**/eval_auto_*.json",
]:
    files = sorted(glob.glob(pat, recursive=True))
    for jp in files:
        try:
            d = json.load(open(jp))
            print(f"{jp.split('/')[-3]}: step {d.get('step')} mse={d.get('mse'):.4f} ssim={d.get('ssim'):.4f}", flush=True)
        except Exception as ex:
            print(jp, "ERR", ex, flush=True)