# -*- coding: utf-8 -*-
"""verify_aug_poster.py — 逐格核对海报内容与源图一致（我这个模型看不了图, 改用数值核对）"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IDX = [131, 100, 86, 67, 30, 177, 173, 22]
COLS = ["gt", "arm4_noaug", "arm2_sym3x", "A1_morphE", "A2_elastic", "A3_grid", "A4_affine"]
CELL, PADL, HDR = 118, 130, 96
P = np.asarray(Image.open("ANALYSIS/aug_family_100k.png").convert("L"))
print("  海报尺寸:", P.shape)

bad, tot, stds = 0, 0, []
for i, idx in enumerate(IDX):
    for j, key in enumerate(COLS):
        y, x = HDR + i * CELL, PADL + j * CELL
        cell = P[y:y + CELL, x:x + CELL]
        stds.append(float(np.std(cell)))
        pre = "" if key == "gt" else key + "/"
        src = f"_local_ckpt/aug_all/{pre}{'gt' if key == 'gt' else 'g'}{idx}.png"
        if not os.path.exists(src):
            print(f"  ✗ 缺源图 {src}")
            continue
        s = np.asarray(Image.open(src).convert("L").resize((CELL, CELL)))
        tot += 1
        d = float(np.abs(cell[10:-10, 10:-10].astype(float) - s[10:-10, 10:-10].astype(float)).mean())
        if d > 2.0:
            bad += 1
            print(f"  ✗ 第{i}行 {key} diff={d:.2f}")
print(f"  逐格比对: {tot - bad}/{tot} 格与源图一致(中心区 diff<=2) " +
      ("✓ 海报内容正确" if bad == 0 and tot > 0 else "✗ 有异常"))
print(f"  每格 std 范围: {min(stds):.1f} ~ {max(stds):.1f}  (≈0 即为空白)")
print(f"  空白格数(std<3): {sum(1 for s in stds if s < 3)}/{len(stds)}")