#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("assets/train_top10_style23.csv")
print(f"总样本数: {len(df)}")
print(f"槽位数: {df['pair_id'].nunique()}")
print("\n各槽位统计:")
for pid in sorted(df["pair_id"].unique()):
    sub = df[df["pair_id"] == pid]
    sname = sub["slot_name"].iloc[0]
    print(f"  slot {pid:>2}: {sname:<10} -> {len(sub):>5} 张 (最小图: {sub['image_path'].iloc[0]})")
