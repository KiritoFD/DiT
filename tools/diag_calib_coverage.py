# -*- coding: utf-8 -*-
"""diag_calib_coverage.py — 定准实验前的覆盖探测: 各 eval id 能否拿到 g_gt 与 g_std。

只读。检查:
  seen20  : g_gt <- shards_aux_skel3 (训练 GT 骨架)   | g_std <- shards_std
  strict84: g_gt <- gt_skel_eval_strict84             | g_std <- shards_std / 50k_v2 shards_std
并报告 eval csv 的 std_path 是否可用作兜底。
"""
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

B = "data/top10_style23"
GT_TRAIN = f"{B}/shards_aux_skel3"        # 训练 GT 骨架
STD = f"{B}/shards_std"                   # 标准字骨架 (训练 id)
GT_STRICT = f"{B}/gt_skel_eval_strict84"  # strict 的 GT 骨架
STD_ALT = "data/50k_v2_glyph15k/shards_std"


def ids_of(d):
    s = set()
    if not os.path.isdir(d):
        return s
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            s.update(int(i) for i in z["img_ids"])
    return s


def csv_ids(p):
    ids = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            ids.append((int(m.group(1)), r))
    return ids


gt_tr, std_s, gt_st, std_alt = (ids_of(x) for x in
                                (GT_TRAIN, STD, GT_STRICT, STD_ALT))
print(f"{GT_TRAIN}: {len(gt_tr)} ids")
print(f"{STD}: {len(std_s)} ids")
print(f"{GT_STRICT}: {len(gt_st)} ids")
print(f"{STD_ALT}: {len(std_alt)} ids\n")

for name, p, gtset in (("seen20", "assets/eval_top10_seen_20.csv", gt_tr),
                       ("strict84", "assets/eval_top10_strict_subset84.csv", gt_st)):
    if not os.path.exists(p):
        print(f"{name}: csv 不存在")
        continue
    items = csv_ids(p)
    ids = {i for i, _ in items}
    print(f"=== {name}: {len(ids)} ids ===")
    print(f"  g_gt  命中 {len(ids & gtset)}/{len(ids)}")
    print(f"  g_std 命中 {len(ids & std_s)}/{len(ids)}  (shards_std)")
    if std_alt:
        print(f"  g_std 命中 {len(ids & std_alt)}/{len(ids)}  ({STD_ALT})")
    have = sum(1 for _, r in items if os.path.exists(r.get("std_path", "")))
    print(f"  csv.std_path 文件存在 {have}/{len(items)}")
    if have:
        print(f"    例: {items[0][1].get('std_path')}")
    print()
