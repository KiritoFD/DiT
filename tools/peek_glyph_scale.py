#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""peek_glyph_scale.py — 读多个 ckpt 的 glyph_scale 学习值 (快速对照 g 通路是否激活)."""
import sys, glob, torch

paths = {
    "v10b-GT-g(85000)": "assets/results/v10b_skel_only_pretrain/*/checkpoints/0085000.pt",
    "v10b-stdskel-pretrain": "assets/results/v10b_stdskel_pretrain/*/checkpoints/*.pt",
    "fame3(57500)": "assets/results/v10b_stdskel_fame3/*/checkpoints/0057500.pt",
}
for name, pat in paths.items():
    fs = sorted(glob.glob(pat), key=lambda p: int(p.split("/")[-1].split(".")[0]))
    if not fs:
        print(f"{name}: NO CKPT")
        continue
    p = fs[-1]
    step = p.split("/")[-1].split(".")[0]
    d = torch.load(p, map_location="cpu", weights_only=False)
    sd = d.get("ema") or d.get("model") or d
    hit = []
    for k, v in sd.items():
        if "glyph_scale" in k:
            hit.append((k, round(float(v), 5)))
    print(f"{name} step={step}: {hit}")