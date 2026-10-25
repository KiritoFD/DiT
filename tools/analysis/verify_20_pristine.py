import csv
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import label

# Select 20 best, pristine, uncropped samples from the 31 clean common samples
# 5 from each of the 4 groups:
# Group 1: 王羲之 楷书
# Group 2: 王羲之 行书
# Group 3: 苏轼 楷书
# Group 4: 苏轼 行书

candidates_pool = [
    # 组 1: 王羲之 楷书 (available: 080 旨, 081 尚, 082 位, 084 兩, 085 咽, 087 卅)
    {"i2": 0,  "i5": 18, "i10": 80,  "callig": "王羲之", "script": "楷", "char": "旨", "group": "王羲之 · 楷书"},
    {"i2": 1,  "i5": 19, "i10": 81,  "callig": "王羲之", "script": "楷", "char": "尚", "group": "王羲之 · 楷书"},
    {"i2": 2,  "i5": 20, "i10": 82,  "callig": "王羲之", "script": "楷", "char": "位", "group": "王羲之 · 楷书"},
    {"i2": 4,  "i5": 22, "i10": 84,  "callig": "王羲之", "script": "楷", "char": "兩", "group": "王羲之 · 楷书"},
    {"i2": 7,  "i5": 25, "i10": 87,  "callig": "王羲之", "script": "楷", "char": "卅", "group": "王羲之 · 楷书"},

    # 组 2: 王羲之 行书 (available: 088 好, 089 致, 090 夜, 091 羲, 092 觴, 093 面, 094 閒, 095 丸, 096 将)
    {"i2": 25, "i5": 26, "i10": 88,  "callig": "王羲之", "script": "行", "char": "好", "group": "王羲之 · 行书"},
    {"i2": 26, "i5": 27, "i10": 89,  "callig": "王羲之", "script": "行", "char": "致", "group": "王羲之 · 行书"},
    {"i2": 27, "i5": 28, "i10": 90,  "callig": "王羲之", "script": "行", "char": "夜", "group": "王羲之 · 行书"},
    {"i2": 28, "i5": 29, "i10": 91,  "callig": "王羲之", "script": "行", "char": "羲", "group": "王羲之 · 行书"},
    {"i2": 33, "i5": 34, "i10": 96,  "callig": "王羲之", "script": "行", "char": "将", "group": "王羲之 · 行书"},

    # 组 3: 苏轼 楷书 (available: 111 靑, 112 衆, 113 六, 114 照, 115 入, 116 土, 117 皇, 119 目)
    {"i2": 50, "i5": 35, "i10": 111, "callig": "苏轼",   "script": "楷", "char": "靑", "group": "苏轼 · 楷书"},
    {"i2": 51, "i5": 36, "i10": 112, "callig": "苏轼",   "script": "楷", "char": "衆", "group": "苏轼 · 楷书"},
    {"i2": 53, "i5": 38, "i10": 114, "callig": "苏轼",   "script": "楷", "char": "照", "group": "苏轼 · 楷书"},
    {"i2": 55, "i5": 40, "i10": 116, "callig": "苏轼",   "script": "楷", "char": "土", "group": "苏轼 · 楷书"},
    {"i2": 56, "i5": 41, "i10": 117, "callig": "苏轼",   "script": "楷", "char": "皇", "group": "苏轼 · 楷书"},

    # 组 4: 苏轼 行书 (available: 121 巫, 122 稜, 123 渚, 124 冒, 125 狐, 126 怆, 127 桑)
    {"i2": 76, "i5": 45, "i10": 121, "callig": "苏轼",   "script": "行", "char": "巫", "group": "苏轼 · 行书"},
    {"i2": 77, "i5": 46, "i10": 122, "callig": "苏轼",   "script": "行", "char": "稜", "group": "苏轼 · 行书"},
    {"i2": 78, "i5": 47, "i10": 123, "callig": "苏轼",   "script": "行", "char": "渚", "group": "苏轼 · 行书"},
    {"i2": 79, "i5": 48, "i10": 124, "callig": "苏轼",   "script": "行", "char": "冒", "group": "苏轼 · 行书"},
    {"i2": 81, "i5": 50, "i10": 126, "callig": "苏轼",   "script": "行", "char": "怆", "group": "苏轼 · 行书"},
]

print("Verifying the 20 selected clean samples:")
for c in candidates_pool:
    p_gt = Path(f"artifacts/DiC_U100_review_20261020/images/reference/gt/g{c['i10']}.png")
    im = Image.open(p_gt).convert("L")
    arr = np.array(im)
    ink = arr < 128
    lbl, num = label(ink)
    sizes = np.bincount(lbl.ravel())[1:] if num > 0 else []
    debris = sum(s < 40 for s in sizes)
    print(f"{c['group']} 【{c['char']}】 (GT g{c['i10']:03d}): components={num:2d}, debris={debris:1d}, ink_area={int(ink.sum()):5d}")
