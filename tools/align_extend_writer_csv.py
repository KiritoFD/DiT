#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/align_extend_writer_csv.py — 校验并对齐已落盘的 5,802 张图片与骨架"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = "/root/Workspace/xy/DiT"
CSV_PATH = os.path.join(ROOT, "assets/train_extend_writer.csv")

def main():
    df = pd.read_csv(CSV_PATH)
    valid_rows = []
    for _, r in df.iterrows():
        img_ok = os.path.exists(os.path.join(ROOT, r["image_path"]))
        std_ok = os.path.exists(os.path.join(ROOT, r["std_path"]))
        if img_ok and std_ok:
            valid_rows.append(r)

    df_clean = pd.DataFrame(valid_rows)
    df_clean.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"🎉 已100%精确对齐落盘文件！样本数: {len(df_clean)} 条，写入: {CSV_PATH}")

if __name__ == "__main__":
    main()
