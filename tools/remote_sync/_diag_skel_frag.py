# -*- coding: utf-8 -*-
"""盲检: 远程原始 final_skeleton 某字的连通分量 vs 降采样产物。
判断 GT骨架碎是原始问题还是降采样导致。"""
import numpy as np, os
from PIL import Image
from scipy.ndimage import label

def stat(path, tag):
    if not os.path.exists(path):
        print(f"[{tag}] 缺 {path}"); return
    a = np.asarray(Image.open(path).convert("L"))
    bw = (a>64)
    n = label(bw)[1]
    print(f"[{tag}] {os.path.basename(path)}: shape={a.shape} 墨像素={int(bw.sum())} 连通分量={n}")

# 找 '一' 的一个样本 (隶书相关)
import csv
rows = list(csv.DictReader(open("5script/train.csv", encoding="utf-8")))
for r in rows[:20000]:
    if r["character"]=="一" and r["script"]=="隶":
        imgid=os.path.basename(r["image_path"])[:-4]
        stat(f"final_skeleton/{imgid}.png","原始骨架")
        stat(f"final_canny/{imgid}.png","原始canny")
        break
