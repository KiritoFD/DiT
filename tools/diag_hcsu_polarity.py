#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_hcsu_polarity.py — 验证 hcsu_wild_salvage 子集的极性反转假设。"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
df = pd.read_csv("assets/train_top10_style23.csv")

sub = df[df["source"] == "hcsu_wild_salvage"].copy()
print("hcsu_wild_salvage 样本数:", len(sub))
print("idx 范围:", sub.index.min(), "..", sub.index.max())

rows = []
rng = np.random.default_rng(0)
sel = sub if len(sub) <= 300 else sub.iloc[rng.choice(len(sub), 300, replace=False)]

for _, r in sel.iterrows():
    tp = r["image_path"]; sp = str(r["src_image_path"])
    if not (os.path.exists(tp) and os.path.exists(sp)):
        continue
    a = np.asarray(Image.open(tp).convert("L").resize((64, 64), Image.Resampling.LANCZOS)).astype(np.float32) / 255.
    b = np.asarray(Image.open(sp).convert("L").resize((64, 64), Image.Resampling.LANCZOS)).astype(np.float32) / 255.
    # 源图暗像素占比(可能是黑底白字)
    src_dark = float((b < 0.5).mean())
    src_bright = float((b > 0.5).mean())
    dst_ink = float((a < 0.5).mean())
    rows.append((src_dark, src_bright, dst_ink))

rows = np.array(rows)
print()
print("采样样本数:", len(rows))
print("源图 暗像素占比(<0.5) mean=%.3f median=%.3f" % (rows[:,0].mean(), np.median(rows[:,0])))
print("源图 亮像素占比(>0.5) mean=%.3f median=%.3f" % (rows[:,1].mean(), np.median(rows[:,1])))
print("数据集 墨迹占比(<0.5) mean=%.3f median=%.3f" % (rows[:,2].mean(), np.median(rows[:,2])))
print()
# 分组
blackbg = rows[rows[:,0] > 0.8]   # 源图黑底
whitebg = rows[rows[:,0] < 0.2]   # 源图白底
print("源图为黑底(暗>80%%)的样本: %d 条 -> 数据集墨迹占比 mean=%.4f" % (len(blackbg), blackbg[:,2].mean() if len(blackbg) else -1))
print("源图为白底(暗<20%%)的样本: %d 条 -> 数据集墨迹占比 mean=%.4f" % (len(whitebg), whitebg[:,2].mean() if len(whitebg) else -1))
print()
print("[结论] 若黑底样本的墨迹占比≈0 -> 转换时未反相, 黑底白字被当成全墨/全白丢失")
