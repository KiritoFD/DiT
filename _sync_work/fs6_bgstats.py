#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_bgstats.py — 50k 训练图 vs 各 few-shot 主题图的底色/对比度分布，判断要不要做自动电平。

Diff loss 的绝对值受"背景灰度/对比度"影响很大：如果主题图是灰底宣纸而 50k 是接近纯白,
那 few-shot 的 loss 高就有一部分来自**底色**而不是**风格**, 判读会错。
"""
import glob
import os

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None


def stats(paths, label):
    bg, ink, gm = [], [], []
    for p in paths[::max(1, len(paths) // 400)]:
        try:
            a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        except Exception:
            continue
        gm.append(a.mean())
        bg.append(np.percentile(a, 90))          # 背景 = 亮端 90 分位
        ink.append((a < 128).mean())
    gm, bg, ink = np.array(gm), np.array(bg), np.array(ink)
    print(f"{label:<16} n={len(gm):>4}  灰度均值={gm.mean():6.1f}  "
          f"底色p90={bg.mean():6.1f}±{bg.std():4.1f}  墨占比={ink.mean():.3f}")


k50 = sorted(glob.glob("data/50k/imgs/*.png"))
stats(k50, "50k 训练图")
for t in ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶", "宋高宗-楷", "徐渭-行"]:
    stats(sorted(glob.glob(f"data/50k/fs6_imgs_{t}/*.png")), t)
stats([r for r in sorted(glob.glob("data/50k/std/*.png"))], "std 骨架图")
