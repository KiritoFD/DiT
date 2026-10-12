#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""settle_v81.py — 定案：保存的 g#.png 与 gt#.png 的 SSIM, 与评测打印的 ssim 比对

  若 saved-ssim ≈ 0.15（空白 vs GT）而打印 ssim=0.63  -> **保存的图是错张量**（评测对但落盘错）
  若 saved-ssim ≈ 0.63                              -> **打印的指标错**（落盘对但指标错）
"""
import glob
import os
import sys

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

os.chdir("/home/ds/Workspace/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 评测打印值（生成时同口径输出）
PRINTED = {"0020000": 0.5549, "0035000": 0.5512, "0040000": 0.5541, "0050000": 0.5771,
           "0055000": 0.6075, "0100000": 0.6302, "0110000": 0.6170, "0115000": 0.6284}

print(f"  {'ckpt':>8} | {'打印 ssim':>9} | {'saved g# vs gt#':>15} | 判定")
for st, printed in PRINTED.items():
    root = f"_local_ckpt/gen_v81sw_{st}_n10/eval_samples_ctrl/step0000000/genrows"
    gs = sorted(glob.glob(os.path.join(root, "g[0-9]*.png")))
    ts = sorted(glob.glob(os.path.join(root, "gt[0-9]*.png")))
    if not gs or not ts:
        print(f"  {st:>8} | {printed:>9.4f} | {'缺文件':>15} |")
        continue
    vals = []
    for a, b in zip(gs, ts):
        A = np.asarray(Image.open(a).convert("RGB").resize((256, 256)))
        B = np.asarray(Image.open(b).convert("RGB").resize((256, 256)))
        vals.append(ssim_fn(A, B, channel_axis=2, data_range=255))
    m = float(np.mean(vals))
    verdict = ("保存图=错张量?" if abs(m - printed) > 0.15 else "一致 ✓")
    print(f"  {st:>8} | {printed:>9.4f} | {m:>15.4f} | {verdict}")
