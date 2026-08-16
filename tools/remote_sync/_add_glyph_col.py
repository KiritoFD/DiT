# -*- coding: utf-8 -*-
"""给 5script 的 csv 加 glyph_id 列：glyph_id = script_id * num_characters + character_id
（V3-A 二因子化：script×char 合并为一个 glyph 类，作为 2cond 模型的 y_char）。"""
import csv
import sys
import glob
import os

NUM_CHARACTERS = 7026  # 与 exp 配置 num_characters 一致（每 script 的字符数）

def add_glyph_col(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            rows.append(row)
    if "glyph_id" in fields:
        print(f"[skip] {path} already has glyph_id")
        return
    new_fields = fields + ["glyph_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            row["glyph_id"] = int(row["script_id"]) * NUM_CHARACTERS + int(row["character_id"])
            writer.writerow(row)
    print(f"[ok] {path}: {len(rows)} rows -> glyph_id added")

if __name__ == "__main__":
    targets = []
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = ["5script/train.csv", "5script/test.csv", "5script/eval.csv",
                   "5script/fast100.csv"] + sorted(glob.glob("5script/eval_strata/*.csv"))
    for t in targets:
        if os.path.isfile(t):
            add_glyph_col(t)
        else:
            print(f"[missing] {t}")
