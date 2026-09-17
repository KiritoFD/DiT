# -*- coding: utf-8 -*-
"""dump_cells.py — 把 clean montage 里指定位置的原图导出, 核实是否真有黑块。"""
import csv
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = "/root/Workspace/xy/DiT/_otout_cells"
os.makedirs(OUT, exist_ok=True)

TH = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
d = np.load("/tmp/scan_metrics.npz")
w, inn, ink, blob = d["w_max"], d["inner"], d["ink"], d["blob"]
rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
clean = np.nonzero(w <= TH)[0]
print(f"clean={len(clean)}  th={TH}")

# 导出 clean 的前 48 个 cell 的原尺寸图, 文件名带指标
for i, k in enumerate(clean[:48]):
    p = rows[int(k)]["image_path"]
    if not os.path.exists(p):
        continue
    dst = os.path.join(OUT, f"{i:02d}_w{w[k]:.0f}_in{inn[k]:.2f}_"
                            f"{rows[int(k)].get('character','')}.png")
    shutil.copy(p, dst)
print(f"-> {OUT}")

# 额外: 找 一/七 字里 w 最大的 10 张 (它们 w<=60 才在 clean 里)
for ch in ("一", "七", "二"):
    idx = [int(k) for k in clean if rows[int(k)].get("character") == ch]
    if not idx:
        continue
    idx.sort(key=lambda k: -w[k])
    print(f"\n  clean 里 '{ch}' 共 {len(idx)} 张, w 最大的 8 张:")
    for k in idx[:8]:
        r = rows[k]
        print(f"    {r['image_path']} w={w[k]:.1f} inner={inn[k]:.2f} "
              f"ink={ink[k]:.3f} blob={blob[k]:.3f}")
