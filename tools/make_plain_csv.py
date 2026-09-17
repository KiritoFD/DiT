# -*- coding: utf-8 -*-
"""make_plain_csv.py — 生成"只用原图、不含任何增强"的训练 csv.

用户裁定 (2026-09-15): 先排除增强带来的所有变量, 用清洗后干净的原图起一个
预训练, 先把"正确性"立住。

输入: assets/train_base_clean.csv         (40,606 行, 已清洗的原图)
输出: assets/train_base_clean_plain.csv   (同上, 但只含 aug=="" 的行)
说明: 该 csv 的 img_id 都是**原始 id**, 其 latent / skel / dino 均已验证正确。
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "assets/train_base_clean.csv"
DST = "assets/train_base_clean_plain.csv"

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
print(f"[plain] {SRC}: {len(rows)} 行")
aug = [r for r in rows if (r.get("aug") or "") != ""]
print(f"  其中带 aug 的行: {len(aug)}")
keep = [r for r in rows if (r.get("aug") or "") == ""]
if not keep:
    keep = rows            # 该 csv 本就只有原图
print(f"  保留 {len(keep)} 行")

fields = list(rows[0].keys())
with open(DST, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in keep:
        r0 = dict(r)
        r0["aug"] = ""
        w.writerow(r0)
print(f"  -> {DST}")

# 检查 id 唯一性
ids = [os.path.basename(r["image_path"])[:-4] for r in keep]
print(f"  id 唯一性: {len(ids)} 行, {len(set(ids))} 个唯一 id")
from collections import Counter
c = Counter(ids).most_common(3)
print(f"  最常见 id 重复: {c}")
