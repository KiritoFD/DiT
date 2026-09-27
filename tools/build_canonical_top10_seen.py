#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/build_canonical_top10_seen.py — 从 train_top10_style23.csv 中按书家均匀抽取 20 张 Seen 评测集"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_CSV = "assets/train_top10_style23.csv"
OUT_SEEN = "assets/eval_top10_seen_20.csv"

TOP10_CALS = ["王羲之", "苏轼", "赵孟頫", "欧阳询", "颜真卿", "褚遂良", "米芾", "柳公权", "何绍基", "文徵明"]

def main():
    df = pd.read_csv(TRAIN_CSV)
    seen_rows = []
    for c in TOP10_CALS:
        sub = df[df["calligrapher"] == c]
        if len(sub) >= 2:
            picked = sub.sample(2, random_state=42)
            seen_rows.append(picked)

    df_seen = pd.concat(seen_rows, ignore_index=True)
    df_seen.to_csv(OUT_SEEN, index=False, encoding="utf-8")
    print(f"🎉 规范的 eval_top10_seen_20.csv 已生成: {OUT_SEEN} ({len(df_seen)} 行)")
    print(df_seen[["img_id", "calligrapher", "script", "character", "image_path"]].to_string())

if __name__ == "__main__":
    main()
