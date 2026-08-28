# -*- coding: utf-8 -*-
"""
build_mid_common_csv.py — 从 train_mid_clean.csv 筛出与 final_imgs_256 重叠的 id,
生成 train_mid_common.csv (mid-clean ∩ 原始图集 = 未增强子集).
"""
import os
import csv
import re

SRC_CSV = "5script/train_mid_clean.csv"
IMG_256 = "final_imgs_256"
OUT_CSV = "5script/train_mid_common.csv"


def main():
    have = set()
    for fn in os.listdir(IMG_256):
        if fn.endswith(".png"):
            have.add(fn[:-4])
    print(f"[mid-common] final_imgs_256 files: {len(have)}")

    rows = []
    kept = 0
    skipped = 0
    with open(SRC_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            m = re.search(r"(\d+)\.png", row["image_path"])
            if m and m.group(1) in have:
                rows.append(row)
                kept += 1
            else:
                skipped += 1

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[mid-common] kept={kept} skipped(不见于原始图集)={skipped} -> {OUT_CSV}")


if __name__ == "__main__":
    main()