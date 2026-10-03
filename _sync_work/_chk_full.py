# -*- coding: utf-8 -*-
"""_chk_full.py — 检查 std_skel3_latents_base_full 的 img_id 覆盖, 并验证按
(script, character) 复用能否覆盖 158K 全量."""
import csv
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(d):
    m = {}
    for sp in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(sp) as z:
            ids = z["img_ids"]
            for j, i in enumerate(ids):
                m[int(i)] = j
    return m, len(m)


for d in ("data/skel/std_skel3_latents_base_full",
          "data/skel/std_skel3_latents_base",
          "data/skel/std_skel1_latents_fame_e"):
    if os.path.isdir(d):
        m, n = load(d)
        print(f"{d}: {len(glob.glob(os.path.join(d,'*.npz')))} npz -> {n} ids")

# base_full 覆盖 train_base_noaug 的哪些行?
rows0 = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
m_full, n_full = load("data/skel/std_skel3_latents_base_full")
hit0 = sum(1 for r in rows0
           if int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) in m_full)
print(f"\nbase_full 覆盖 base_noaug: {hit0}/{len(rows0)} = {100*hit0/len(rows0):.2f}%")

# (script, character) -> 原图 img_id (从原图行建)
ch2iid = {}
for r in rows0:
    iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
    if iid in m_full:
        ch2iid.setdefault((r["script"], r["character"]), iid)
print(f"可复用的 (script,character) -> 原图: {len(ch2iid)}")

# 158K 全量覆盖?
rows = list(csv.DictReader(open("assets/train_base_sym.csv", encoding="utf-8")))
miss = Counter()
n_ok = 0
for r in rows:
    m = re.search(r"(\d+)\.png", r["image_path"])
    iid = int(m.group(1)) if m else -1
    if iid in m_full:
        n_ok += 1
        continue
    k = (r.get("script", ""), r.get("character", ""))
    if k in ch2iid:
        n_ok += 1
        continue
    miss[r.get("image_path", "").split("/")[2] if len(r.get("image_path", "").split("/")) > 2 else "?"] += 1
print(f"158K 覆盖 (直接 img_id 或 按字复用): {n_ok}/{len(rows)} = {100*n_ok/len(rows):.2f}%")
print(f"  仍缺: {miss.most_common()}")
