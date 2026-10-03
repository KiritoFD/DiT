# -*- coding: utf-8 -*-
"""_diag_blackblob.py — 定位 seen 里"多"字生成黑块的来源.

假设方向:
  A) 训练集里该字的某些样本本身就是"墨块"(增强 dilate 把笔画连成一片 / 扫描阴影 / 坏图)
     -> 模型在 seen 上学到并复现
  B) g 条件(标准字骨架)异常
  C) GT 本身异常
"""
import csv
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET = sys.argv[1] if len(sys.argv) > 1 else "多"
SEEN = "assets/eval_seen_v10.csv"
TRAIN = "assets/train_base_sym.csv"


def fg_ratio(p, thr=127):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    return (a < thr).mean(), a


print(f"=== 1) seen 集里的 '{TARGET}' ===")
rows = list(csv.DictReader(open(SEEN, encoding="utf-8")))
hit = [r for r in rows if r.get("character") == TARGET]
print(f"  seen n={len(rows)}, 命中 '{TARGET}' {len(hit)} 个")
for i, r in enumerate(hit):
    print(f"    idx={i} callig={r.get('calligrapher')} script={r.get('script')} "
          f"img={r.get('image_path')}")

print(f"\n=== 2) 训练集里 '{TARGET}' 的所有样本 ===")
tr = [r for r in csv.DictReader(open(TRAIN, encoding="utf-8"))
      if r.get("character") == TARGET]
print(f"  训练行数={len(tr)}  aug 分布={Counter(r.get('aug', '') for r in tr).most_common()}")

# 检查训练图的"墨量", 找出异常大块的
sus = []
for r in tr:
    p = r.get("image_path", "")
    if not os.path.exists(p):
        continue
    g, a = fg_ratio(p)
    if g > 0.35:                      # 前景占比 >35% 视为可疑墨块
        sus.append((g, r.get("aug", ""), r.get("calligrapher"), p))
sus.sort(reverse=True)
print(f"  前景占比>35% 的可疑墨块: {len(sus)}")
for g, aug, cal, p in sus[:12]:
    print(f"    fg={g:.3f} aug={aug:3s} callig={cal:8s} {p}")

# 统计整体分布
allg = []
for r in tr:
    p = r.get("image_path", "")
    if os.path.exists(p):
        g, _ = fg_ratio(p)
        allg.append(g)
if allg:
    allg = np.array(allg)
    print(f"\n  训练图前景占比: p50={np.percentile(allg,50):.3f} "
          f"p90={np.percentile(allg,90):.3f} p99={np.percentile(allg,99):.3f} "
          f"max={allg.max():.3f}")

print(f"\n=== 3) 该字的 g 条件(标准字骨架) ===")
key2uid = {}
for r in csv.DictReader(open("data/skel/std_skel3_base_key2uid.csv", encoding="utf-8")):
    key2uid[(r["script"], r["character"])] = int(r["uid"])
for sc in ("楷", "行", "隶"):
    u = key2uid.get((sc, TARGET))
    if u is not None:
        p = f"data/skel/std_skel3_base_png/{u}.png"
        if os.path.exists(p):
            g, _ = fg_ratio(p)
            print(f"    script={sc} uid={u} fg={g:.3f}  {p}")
        else:
            print(f"    script={sc} uid={u} (PNG 不存在: {p})")
    else:
        print(f"    script={sc} (key2uid 无此组合)")
