import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, ".")
from tools.render_fixed_20row_poster import measure_p50_and_contrast
from tools.analysis.test_fixed_powerful import test_fixed_powerful_align

with open("assets/stroke_width_trainset_distribution.json", encoding="utf-8") as f:
    train_dist = json.load(f)

# The 20 samples (with g115 instead of g114)
SAMPLES = [
    {"i2": 0,  "i5": 18, "i10": 80,  "callig": "王羲之", "script": "楷", "char": "旨", "group": "王羲之 · 楷书"},
    {"i2": 1,  "i5": 19, "i10": 81,  "callig": "王羲之", "script": "楷", "char": "尚", "group": "王羲之 · 楷书"},
    {"i2": 2,  "i5": 20, "i10": 82,  "callig": "王羲之", "script": "楷", "char": "位", "group": "王羲之 · 楷书"},
    {"i2": 4,  "i5": 22, "i10": 84,  "callig": "王羲之", "script": "楷", "char": "兩", "group": "王羲之 · 楷书"},
    {"i2": 7,  "i5": 25, "i10": 87,  "callig": "王羲之", "script": "楷", "char": "卅", "group": "王羲之 · 楷书"},

    {"i2": 25, "i5": 26, "i10": 88,  "callig": "王羲之", "script": "行", "char": "好", "group": "王羲之 · 行书"},
    {"i2": 26, "i5": 27, "i10": 89,  "callig": "王羲之", "script": "行", "char": "致", "group": "王羲之 · 行书"},
    {"i2": 27, "i5": 28, "i10": 90,  "callig": "王羲之", "script": "行", "char": "夜", "group": "王羲之 · 行书"},
    {"i2": 28, "i5": 29, "i10": 91,  "callig": "王羲之", "script": "行", "char": "羲", "group": "王羲之 · 行书"},
    {"i2": 33, "i5": 34, "i10": 96,  "callig": "王羲之", "script": "行", "char": "将", "group": "王羲之 · 行书"},

    {"i2": 50, "i5": 35, "i10": 111, "callig": "苏轼",   "script": "楷", "char": "靑", "group": "苏轼 · 楷书"},
    {"i2": 51, "i5": 36, "i10": 112, "callig": "苏轼",   "script": "楷", "char": "衆", "group": "苏轼 · 楷书"},
    {"i2": 54, "i5": 39, "i10": 115, "callig": "苏轼",   "script": "楷", "char": "入", "group": "苏轼 · 楷书"},
    {"i2": 55, "i5": 40, "i10": 116, "callig": "苏轼",   "script": "楷", "char": "土", "group": "苏轼 · 楷书"},
    {"i2": 56, "i5": 41, "i10": 117, "callig": "苏轼",   "script": "楷", "char": "皇", "group": "苏轼 · 楷书"},

    {"i2": 76, "i5": 45, "i10": 121, "callig": "苏轼",   "script": "行", "char": "巫", "group": "苏轼 · 行书"},
    {"i2": 77, "i5": 46, "i10": 122, "callig": "苏轼",   "script": "行", "char": "稜", "group": "苏轼 · 行书"},
    {"i2": 78, "i5": 47, "i10": 123, "callig": "苏轼",   "script": "行", "char": "渚", "group": "苏轼 · 行书"},
    {"i2": 79, "i5": 48, "i10": 124, "callig": "苏轼",   "script": "行", "char": "冒", "group": "苏轼 · 行书"},
    {"i2": 81, "i5": 50, "i10": 126, "callig": "苏轼",   "script": "行", "char": "怆", "group": "苏轼 · 行书"},
]

print(f"{'Group':14s} | {'Char':4s} | {'Target':6s} | {'Raw P50':7s} | {'a=0.35':7s} | {'a=0.70':7s} | {'a=1.00':7s} | {'Change':7s}")
print("-" * 75)

for s in SAMPLES:
    slot = f"{s['callig']}_{s['script']}"
    target = train_dist[slot]
    p_t10 = f"artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g{s['i10']}.png"
    im = np.array(Image.open(p_t10).convert("L"), dtype=np.float32) / 255.0
    
    p50_raw, _ = measure_p50_and_contrast(im)
    res_35 = test_fixed_powerful_align(im, target, alpha=0.35)
    p50_35, _ = measure_p50_and_contrast(res_35)
    res_70 = test_fixed_powerful_align(im, target, alpha=0.70)
    p50_70, _ = measure_p50_and_contrast(res_70)
    res_100 = test_fixed_powerful_align(im, target, alpha=1.00)
    p50_100, _ = measure_p50_and_contrast(res_100)
    
    diff = p50_100 - p50_raw
    grp = s["group"]
    ch = s["char"]
    t_p50 = target["ref_p50"]
    print(f"{grp:14s} | {ch:2s}   | {t_p50:5.1f}px | {p50_raw:6.2f}px | {p50_35:6.2f}px | {p50_70:6.2f}px | {p50_100:6.2f}px | {diff:+6.2f}px")
