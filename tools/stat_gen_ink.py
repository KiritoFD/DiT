# -*- coding: utf-8 -*-
"""stat_gen_ink.py — 统计 eval 生成图的墨占比, 程序化定位"黑块".

黑块特征: 生成图 ink 异常高 (正常书法 3%~45%), 或 RGB 明显偏暗/偏黄。
用法: python tools/stat_gen_ink.py <run_dir>
"""
import glob
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "assets/results/v11_pretrain_Sp2_base_sym_clean"
EVAL = os.path.join(RUN, "eval_samples_ctrl")

print(f"[stat] {EVAL}")
subs = sorted(d for d in os.listdir(EVAL) if d.startswith("step"))
print(f"  Steps: {subs}")

for st in subs[-3:]:
    d = os.path.join(EVAL, st)
    files = sorted(glob.glob(os.path.join(d, "**", "*.png"), recursive=True))
    if not files:
        print(f"  {st}: 无 png")
        continue
    print(f"\n  === {st}: {len(files)} 张 ===")
    subdirs = Counter(os.path.basename(os.path.dirname(f)) for f in files)
    print(f"    子目录: {dict(subdirs)}")
    recs = []
    for f in files:
        try:
            im = Image.open(f).convert("RGB")
            a = np.asarray(im, dtype=np.uint8)
        except Exception:
            continue
        g = np.asarray(im.convert("L"), dtype=np.uint8)
        ink = float((g < 128).mean())
        rgb = a.reshape(-1, 3).mean(axis=0)
        recs.append((f, ink, float(rgb[0] - rgb[2]), rgb))
    if not recs:
        continue
    inks = np.array([r[1] for r in recs])
    rb = np.array([r[2] for r in recs])
    print(f"    ink: p50={np.percentile(inks,50):.3f} p90={np.percentile(inks,90):.3f} "
          f"max={inks.max():.3f}")
    print(f"    R-B(色偏): p50={np.percentile(rb,50):.1f} max={rb.max():.1f}")
    dark = [r for r in recs if r[1] > 0.5]
    print(f"    **ink>0.5 (疑似黑块): {len(dark)} / {len(recs)}**")
    for f, ink, rbv, rgb in dark[:8]:
        print(f"      ink={ink:.3f} R-B={rbv:.1f} RGB={rgb.round(1)} {f[-46:]}")
