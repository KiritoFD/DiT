#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/filter_extend_writer.py — 审计并过滤 extend_writer 集合中的书体与样本"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SALVAGE_CSV = "assets/salvaged_real_calligraphy_6029.csv"
TRAIN_50K = "assets/train_50k_v2_fixed.csv"

def main():
    df_salvage = pd.read_csv(SALVAGE_CSV)
    df_50k = pd.read_csv(TRAIN_50K)

    df_valid = df_salvage[df_salvage["is_salvageable"] == True].copy()
    print(f"初始通过质检的候选真迹: {len(df_valid):,} 张")

    base_cs = df_50k.groupby(["calligrapher", "script"]).size().to_dict()
    salvage_cs = df_valid.groupby(["calligrapher", "script"]).size().to_dict()

    all_pairs = sorted(set(base_cs.keys()) | set(salvage_cs.keys()))
    
    # 规则：如果一个 (书家, 书体) 在原有底库 + 本次打捞后的合并总数仍然 < 30 张，
    # 说明该书家在该书体下依然是“微量毛刺”（如蔡襄-隶共1张，米芾-隶共2张，颜真卿-隶共9张），
    # 坚决剔除；只有合并后总数 >= 30 张的，才算合格的兼擅体或主力体！
    keep_pairs = set()
    drop_pairs = set()

    for c, s in all_pairs:
        b = base_cs.get((c, s), 0)
        v = salvage_cs.get((c, s), 0)
        tot = b + v
        if tot < 30:
            drop_pairs.add((c, s))
        else:
            keep_pairs.add((c, s))

    print(f"\n【微量毛刺书体剔除名单】（合并后总数依然 < 30 张的对）:")
    dropped_rows = 0
    for c, s in sorted(drop_pairs):
        b = base_cs.get((c, s), 0)
        v = salvage_cs.get((c, s), 0)
        if v > 0:
            print(f"  - {c}-{s}: 50k已有 {b} 张, 本次打捞 {v} 张 -> 合计 {b+v} 张 (剔除本次这 {v} 张)")
            dropped_rows += v

    print(f"总计从打捞集中剔除微量毛刺样本: {dropped_rows} 张")

    # 过滤数据
    df_final = df_valid[df_valid.apply(lambda r: (r["calligrapher"], r["script"]) in keep_pairs, axis=1)].copy()
    print(f"\n🎉 最终纯净高质量【extend_writer】样本量: {len(df_final):,} 张！")

    print("\n书体构成:")
    print(df_final["script"].value_counts())

    print("\n主要名家分布 Top 15:")
    print(df_final["calligrapher"].value_counts().head(15))

if __name__ == "__main__":
    main()
