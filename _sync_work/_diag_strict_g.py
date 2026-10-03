# -*- coding: utf-8 -*-
"""_diag_strict_g.py — strict poster 第一行(std)发黄的根因检查.

假设: strict 集的 (script, character) 不在 std skel 的 key2uid 里 -> g 全零
      -> VAE decode(0) = [129,110,89] 灰黄棕 (R-B=+40) -> 看起来"黄底".
"""
import csv
import glob
import os
import re
import sys

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RD = "assets/results/v11_pretrain_Sp2_base_sym/eval_samples_ctrl"

print("=== input_g PNG 颜色体检 ===")
print("  参考: VAE decode(0) = [129,110,89] (灰黄棕, R-B=+40) -> 说明 g 全零")
for sub in ("strict_input_g", "seen_input_g"):
    fs = sorted(glob.glob(os.path.join(RD, sub, "g*.png")))
    if not fs:
        print(f"  [{sub}] 无文件")
        continue
    rows = []
    for p in fs:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        rows.append(a.reshape(-1, 3).mean(0))
    rows = np.array(rows)
    rb = rows[:, 0] - rows[:, 2]
    # "黄/可疑" = 接近 decode(0) 的特征: R-B 大 且 整体偏暗
    susp = ((rb > 25) & (rows.mean(1) < 200)).sum()
    print(f"  [{sub:16s}] n={len(fs)} RGB均值={np.round(rows.mean(0),1)} "
          f"R-B 均值={rb.mean():+.1f}  可疑(发黄)={susp}/{len(fs)}")
    if susp:
        bad = [(os.path.basename(fs[i]), np.round(rows[i], 1)) for i in range(len(fs))
               if (rb[i] > 25 and rows[i].mean() < 200)][:8]
        for nm, v in bad:
            print(f"      黄: {nm}  RGB={v}")

# ---- strict csv 的字符覆盖率 ----
print("\n=== strict 集字符在 std skel key2uid 中的覆盖 ===")
key2uid = {}
for r in csv.DictReader(open("data/skel/std_skel3_base_key2uid.csv", encoding="utf-8")):
    key2uid[(r["script"], r["character"])] = int(r["uid"])
print(f"  base key2uid: {len(key2uid)} 个 (script,character)")

for csvp in ("assets/eval_fame3_strict_clean_v9.csv", "assets/eval_seen_v10.csv"):
    if not os.path.exists(csvp):
        print(f"  {csvp} 不存在")
        continue
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    miss = [(r.get("script", ""), r.get("character", "")) for r in rows
            if (r.get("script", ""), r.get("character", "")) not in key2uid]
    print(f"\n  {csvp}: n={len(rows)}")
    print(f"    列: {list(rows[0].keys())}")
    print(f"    script 取值: {sorted({r.get('script','') for r in rows})}")
    print(f"    未覆盖: {len(miss)}/{len(rows)} = {100*len(miss)/len(rows):.1f}%")
    if miss:
        from collections import Counter
        print(f"    缺失样例(script,char): {Counter(miss).most_common(10)}")
