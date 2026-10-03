# -*- coding: utf-8 -*-
"""_diag_strict_id.py — strict 集 g 全零: 查 img_id 在各 std skel 目录的命中率."""
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ids_of(d):
    s = set()
    for sp in glob.glob(os.path.join(d, "*.npz")):
        with np.load(sp) as z:
            for i in z["img_ids"]:
                s.add(int(i))
    return s


DIRS = [
    "data/skel/std_skel3_latents_base_sym",
    "data/skel/std_skel3_latents_base_full",
    "data/skel/std_skel3_latents_fame_evalext",
    "data/skel/std_skel3_latents_fame_e",
    "data/skel/std_skel3_latents_base",
]
have = {}
for d in DIRS:
    if os.path.isdir(d):
        have[d] = ids_of(d)
        print(f"{d:44s} {len(have[d]):8d} ids")
    else:
        print(f"{d:44s} (不存在)")

for csvp in ("assets/eval_fame3_strict_clean_v9.csv", "assets/eval_seen_v10.csv"):
    if not os.path.exists(csvp):
        continue
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    iids = []
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        iids.append(int(m.group(1)) if m else -1)
    print(f"\n{csvp}: n={len(rows)}  img_id 样例={iids[:6]}  "
          f"范围=[{min(iids)}, {max(iids)}]")
    for d, s in have.items():
        hit = sum(1 for i in iids if i in s)
        print(f"   {d:44s} 命中 {hit:4d}/{len(iids)} = {100*hit/len(iids):5.1f}%")
