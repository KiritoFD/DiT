#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/merge_extend_writer_to_base.py — 合并 extend_writer 到全量训练集 CSV"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_CSV = "assets/train_50k_v2_augmented_glyph15k.csv"
EXT_CSV = "assets/train_extend_writer.csv"
OUT_CSV = "assets/train_50k_v2_augmented_glyph15k_ext.csv"

def main():
    df_base = pd.read_csv(BASE_CSV)
    df_ext = pd.read_csv(EXT_CSV)

    cols = [
        "image_path", "calligrapher", "script", "character",
        "calligrapher_id", "script_id", "character_id", "glyph_id",
        "aug", "std_path", "source", "src_image_path", "old_50k_id"
    ]

    # 对齐列
    if "old_50k_id" not in df_ext.columns and "extend_writer_id" in df_ext.columns:
        df_ext["old_50k_id"] = df_ext["extend_writer_id"]

    for col in cols:
        if col not in df_base.columns:
            df_base[col] = ""
        if col not in df_ext.columns:
            df_ext[col] = ""

    sub_base = df_base[cols]
    sub_ext = df_ext[cols]

    merged = pd.concat([sub_base, sub_ext], ignore_index=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"🎉 全量合并数据集已生成: {OUT_CSV}")
    print(f"  - 基线增广底库: {len(df_base):,} 条")
    print(f"  - extend_writer 增量: {len(df_ext):,} 条")
    print(f"  - 合并后全库规模: {len(merged):,} 条")

if __name__ == "__main__":
    main()
