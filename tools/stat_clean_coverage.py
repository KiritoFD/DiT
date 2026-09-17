# -*- coding: utf-8 -*-
"""stat_clean_coverage.py — 搞清"干净四万张"的规模与标准字(g)覆盖率.

统计:
  1) 行数 / 唯一 img_id / 唯一 character / 唯一 (script,character) / 书家数
  2) 标准字骨架的可用来源与覆盖情况:
     - data/skel/std_skel3_base_png/{uid}.png        (PNG, uid-keyed)
     - data/skel/std_skel3_latents_base_full/*.npz   (latent, img_id-keyed, 合并版)
     - data/skel/std_skel3_latents_base/*.npz        (latent, uid-keyed, 仅新生成)
     - data/skel/std_skel3_latents_fame_sym/*.npz 等
  3) 逐来源判断对该 csv 的覆盖率, 找出没有标准字的 (script,character)
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

CSV = sys.argv[1] if len(sys.argv) > 1 else "assets/train_base_clean.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"[csv] {CSV}: {len(rows)} 行")

ids = [os.path.basename(r["image_path"])[:-4] for r in rows]
chars = Counter(r.get("character", "") for r in rows)
keys = Counter((r.get("script", ""), r.get("character", "")) for r in rows)
cals = Counter(r.get("calligrapher", "") for r in rows)
print(f"  唯一 img_id      : {len(set(ids))}")
print(f"  唯一 character   : {len(chars)}")
print(f"  唯一 (script,char): {len(keys)}")
print(f"  书家数           : {len(cals)}")
print(f"  script 分布      : {dict(Counter(r.get('script','') for r in rows))}")
print(f"  字符出现次数      : p50={sorted(chars.values())[len(chars)//2]} "
      f"min={min(chars.values())} max={max(chars.values())}")

# ---- 标准字来源 ----
print("\n=== 标准字来源 ===")
srcs = {}
p = "data/skel/std_skel3_base_png"
if os.path.isdir(p):
    srcs[p] = ("png-uid", {int(os.path.basename(f)[:-4]) for f in glob.glob(p + "/*.png")})
    print(f"  {p:44s} {len(srcs[p][1])} 个 uid")

for d in ("std_skel3_latents_base_full", "std_skel3_latents_base",
          "std_skel3_latents_fame_sym", "std_skel3_latents_tj",
          "std_skel3_latents_fame_v8", "std_skel3_latents_fame3_v8"):
    p = f"data/skel/{d}"
    if not os.path.isdir(p):
        continue
    s = set()
    for f in sorted(glob.glob(p + "/*.npz")):
        try:
            with np.load(f) as z:
                s.update(int(x) for x in z["img_ids"])
        except Exception:
            pass
    srcs[p] = ("latent-id", s)
    print(f"  {p:44s} {len(s)} 个 img_id")

# key2uid 映射
k2u = {}
p = "data/skel/std_skel3_base_key2uid.csv"
if os.path.exists(p):
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            k2u[(r.get("script", ""), r.get("character", ""))] = int(r["uid"])
        except Exception:
            pass
    print(f"  {p:44s} {len(k2u)} 个 (script,char)->uid")

# ---- 覆盖率 ----
print("\n=== 覆盖率 (对当前 csv) ===")
idset = set(int(x) for x in ids)
for name, (kind, s) in srcs.items():
    if kind == "latent-id":
        cov = len(idset & s)
        print(f"  {name:44s} 直接命中 img_id {cov:6d}/{len(idset)} "
              f"({100*cov/len(idset):.1f}%)")
    else:
        # png 是 uid-keyed, 需要经 key2uid 转换
        hit_ids = {i for i in idset if k2u.get(
            next(((r.get("script", ""), r.get("character", ""))
                  for r in rows if os.path.basename(r["image_path"])[:-4] == str(i)), (None, None))) in s}
        print(f"  {name:44s} 经 key2uid 命中 {len(hit_ids):6d}/{len(idset)} "
              f"({100*len(hit_ids)/len(idset):.1f}%)")

# 哪些 (script,char) 没有标准字
if k2u:
    miss = [k for k in keys if k not in k2u]
    print(f"\n  (script,char) 在 key2uid 中缺失: {len(miss)}/{len(keys)}")
    for k, c in sorted(((k, keys[k]) for k in miss), key=lambda x: -x[1])[:15]:
        print(f"    {k[0]}/{k[1]}  出现 {c} 次")
