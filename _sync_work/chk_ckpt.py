#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看 v31/stage1 ckpt 的结构键, 以便写生成器。"""
import glob
import os

import torch as th

os.chdir("/root/Workspace/xy/DiT")
for pat in ("assets/results/v31_stage1_skel/*/checkpoints/*.pt",
            "assets/results/v32_stage2_img/*/checkpoints/*.pt"):
    fs = sorted(glob.glob(pat))
    if not fs:
        print(pat, "-> 空")
        continue
    p = fs[-1]
    d = th.load(p, map_location="cpu", weights_only=False)
    print(f"--- {p}")
    print("    keys:", list(d.keys()))
    for k, v in d.items():
        if isinstance(v, dict):
            ks = list(v.keys())
            print(f"      {k}: dict[{len(ks)}] 例 {ks[:3]}")
        elif isinstance(v, (str, int, float, bool, type(None))):
            print(f"      {k} = {str(v)[:100]}")
        else:
            print(f"      {k}: {type(v).__name__}")
