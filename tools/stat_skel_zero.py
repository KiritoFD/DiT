# -*- coding: utf-8 -*-
"""stat_skel_zero.py — 统计 g 条件(骨架) latent 中"零向量"的比例.

零 latent 解码出来是灰黄棕色实心块 -> 该样本的 g 条件等于没有, 而 g 是
xattn 全 12 层的主引导信号, 会直接把生成带崩 (糊团/黑块)。
"""
import csv
import glob
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKEL = sys.argv[1] if len(sys.argv) > 1 else "data/skel/std_skel3_latents_base_sym"
CSV = sys.argv[2] if len(sys.argv) > 2 else "assets/train_base_sym_clean.csv"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
want = {int(os.path.basename(r["image_path"])[:-4]) for r in rows}
print(f"[stat] {CSV}: {len(rows)} 行, 唯一 id {len(want)}")
print(f"[stat] skel dir: {SKEL}")

tot = zero = 0
zero_rows = []
by_char = Counter()
allrows = {}
for r in rows:
    allrows[int(os.path.basename(r["image_path"])[:-4])] = r

for sp in sorted(glob.glob(os.path.join(SKEL, "shard_*.npz"))):
    with np.load(sp) as d:
        lat = d["latents"]
        ids = d["img_ids"]
        # 零判定: 全通道均值绝对值极小
        am = np.abs(lat.astype(np.float32)).mean(axis=(1, 2, 3))
        for j, iid in enumerate(ids):
            iid = int(iid)
            if iid not in want:
                continue
            tot += 1
            if am[j] < 0.02:
                zero += 1
                zr = allrows.get(iid)
                if zr:
                    by_char[(zr.get("script", ""), zr.get("character", ""))] += 1
                if len(zero_rows) < 12:
                    zero_rows.append((iid, float(am[j]),
                                      (zr or {}).get("script", ""),
                                      (zr or {}).get("character", ""),
                                      (zr or {}).get("calligrapher", "")))

print(f"\n  参与统计 {tot} 个 id")
print(f"  **零骨架 {zero}  ({100*zero/max(tot,1):.2f}%)**")
print(f"\n  零骨架样例:")
for iid, am, sc, ch, cal in zero_rows:
    print(f"    id={iid} |lat|mean={am:.5f}  {sc}/{ch}  {cal}")
print(f"\n  零骨架按 (script,char) Top15:")
for k, v in by_char.most_common(15):
    print(f"    {k[0]}/{k[1]}  {v}")
print(f"  受影响唯一字符数: {len(by_char)}")
