# -*- coding: utf-8 -*-
"""_diag_dino_dup.py — 查清 dino cache 里 img_id 重复的来源.

DinoFeatureCache 用 img_id 索引 (要求唯一); 报错:
  [dino-cache] duplicate img_ids in data/dino_cache/base_sym_v1/ids.npy
"""
import csv
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 1) cache 自身的重复
D = "data/dino_cache/base_sym_v1"
ids = np.load(os.path.join(D, "ids.npy"))
c = Counter(int(i) for i in ids)
dups = {k: v for k, v in c.items() if v > 1}
print(f"[cache] ids.npy: n={len(ids)} unique={len(c)} 重复 id 个数={len(dups)}")
if dups:
    print(f"  重复次数分布: {Counter(dups.values())}")
    top = sorted(dups.items(), key=lambda x: -x[1])[:8]
    print(f"  重复最多的 id: {top}")

# 2) 上游 csv 的 img_id 重复
for csvp in ("assets/train_base_sym.csv", "assets/train_base_noaug.csv"):
    if not os.path.exists(csvp):
        print(f"\n[{csvp}] 不存在")
        continue
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    cid = Counter()
    for r in rows:
        m = re.search(r"(\d+)\.png", r.get("image_path", ""))
        if m:
            cid[int(m.group(1))] += 1
    d2 = {k: v for k, v in cid.items() if v > 1}
    print(f"\n[{csvp}] rows={len(rows)} unique img_id={len(cid)} 重复 id={len(d2)}")
    if d2:
        print(f"  重复次数分布: {Counter(d2.values())}")
        top = sorted(d2.items(), key=lambda x: -x[1])[:8]
        print(f"  重复最多的 id: {top}")
        # 看这些重复行长什么样
        tgt = {k for k, _ in top[:2]}
        shown = 0
        for r in rows:
            m = re.search(r"(\d+)\.png", r.get("image_path", ""))
            if m and int(m.group(1)) in tgt and shown < 6:
                print(f"    id={m.group(1)} aug={r.get('aug')!r} "
                      f"char={r.get('character')} callig={r.get('calligrapher')} "
                      f"src={r.get('image_path','')[:52]}")
                shown += 1
