#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_how50kwasbuilt.py — 50k 的图相对 wild 原图做了什么处理？(对比同一张图的底色)

若 50k 版本被"洗白"过(底色从灰纸变成纯白)，那 few-shot 也必须做同一道处理，
否则 few-shot 的 Diff/ssim 里混进了**底色差**而不是风格差。
"""
import csv
import os

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None
WILD = "/root/Workspace/xy/HCSU/wild_extract"
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))


def bg(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    return float(np.percentile(a, 90)), float(a.mean()), float((a < 128).mean())


for folder in ["米芾-行", "何绍基-隶", "欧阳询-行", "郑板桥-行", "金农-隶"]:
    picked = [r for r in rows if f"/wild/{folder}/" in (r.get("src_image_path") or "")][:40]
    if not picked:
        print(f"{folder}: CSV 里没找到")
        continue
    d50, draw = [], []
    for r in picked:
        if not os.path.isfile(r["image_path"]):
            continue
        src = os.path.join(WILD, folder, os.path.basename(r["src_image_path"]))
        if not os.path.isfile(src):
            src = os.path.join(WILD, folder, os.path.splitext(os.path.basename(
                r["src_image_path"]))[0] + ".png")
        if not os.path.isfile(src):
            continue
        d50.append(bg(r["image_path"]))
        draw.append(bg(src))
    if not d50:
        print(f"{folder}: 图对不上")
        continue
    a50, aw = np.array(d50), np.array(draw)
    print(f"{folder:<8} n={len(a50):>3}  50k: 底色p90={a50[:,0].mean():6.1f} 灰均值="
          f"{a50[:,1].mean():6.1f} 墨={a50[:,2].mean():.3f}   |   wild 原图: 底色p90="
          f"{aw[:,0].mean():6.1f} 灰均值={aw[:,1].mean():6.1f} 墨={aw[:,2].mean():.3f}")
