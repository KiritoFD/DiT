#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v81_faintness.py — (1) 量化"越训越淡" (2) 用正确的 g#.png 落盘 eval/v81fix_<step>/
"""
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

os.chdir("/home/ds/Workspace/DiT")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEPS = ["0020000", "0035000", "0040000", "0050000", "0055000", "0100000", "0110000", "0115000"]
IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]
rows = list(__import__("csv").DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))


def stats(p):
    a = np.asarray(Image.open(p).convert("L")).astype(np.float32)
    return float(a.mean()), float((a < 128).mean()), float((a < 200).mean())


print("  GT 参照:")
gm, g1, g2 = zip(*[stats(rows[i]["image_path"]) for i in IDX])
print(f"    灰度均值 {np.mean(gm):.1f} | 硬墨(<128) {np.mean(g1):.4f} | 软墨(<200) {np.mean(g2):.4f}")
print()
print(f"  {'ckpt':>8} | {'灰度均值':>8} | {'硬墨<128':>8} | {'软墨<200':>8} | 与 GT 灰度差")
for st in STEPS:
    root = f"_local_ckpt/gen_v81sw_{st}_n10/eval_samples_ctrl/step0000000/genrows"
    gs = sorted(glob.glob(os.path.join(root, "g[0-9]*.png")))
    if len(gs) != len(IDX):
        print(f"  {st:>8} | 缺图 ({len(gs)})")
        continue
    mm, m1, m2 = zip(*[stats(p) for p in gs])
    print(f"  {st:>8} | {np.mean(mm):>8.1f} | {np.mean(m1):>8.4f} | {np.mean(m2):>8.4f} | "
          f"{np.mean(np.array(mm) - np.array(gm)):+.1f}")
    dst = f"eval/v81fix_{st}"
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    for i, p in enumerate(gs):
        shutil.copyfile(p, os.path.join(dst, f"{i}.png"))

DL = " ".join(f"v81fix_{s}" for s in STEPS if os.path.isdir(f"eval/v81fix_{s}"))
os.system(f"rm -f /tmp/v81fix.tgz && tar -czf /tmp/v81fix.tgz -C eval {DL} && ls -la /tmp/v81fix.tgz")
