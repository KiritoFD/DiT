#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/inspect_top10_and_eval.py — 审计 Top 10 书家体系与现有 eval 集兼容性"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_BASE = "assets/train_50k_v2_fixed.csv"
TRAIN_EXT = "assets/train_50k_v2_augmented_glyph15k_ext.csv"
EVAL_STRICT = "assets/eval_v13_strict_fixed.csv"
EVAL_SEEN = "assets/eval_v13_seen.csv"

def main():
    df_train = pd.read_csv(TRAIN_EXT)
    df_eval_strict = pd.read_csv(EVAL_STRICT)
    df_eval_seen = pd.read_csv(EVAL_SEEN)

    print("=" * 70)
    print("Top 10 书家体系构建与当前 Eval 集兼容性审计")
    print("=" * 70)

    # 1. 统计当前全量训练集（72k）中书家样本量排行
    cal_counts = df_train["calligrapher"].value_counts()
    top10_cals = list(cal_counts.index[:10])
    
    print("\n【1. 全量训练集中样本量最大的 Top 10 书家】:")
    for i, c in enumerate(top10_cals):
        cnt = cal_counts[c]
        print(f"  {i+1:>2}. {c:<8}: {cnt:>5} 张 ({cnt/len(df_train)*100:.1f}%)")

    top10_total_samples = sum(cal_counts[c] for c in top10_cals)
    print(f"\nTop 10 书家累计样本量: {top10_total_samples:,} 张 (占全库 72k 的 {top10_total_samples/len(df_train)*100:.1f}%)")

    # 2. 检查 Top 10 的 (书家 x 书体) 细分槽位数量
    df_top10 = df_train[df_train["calligrapher"].isin(top10_cals)]
    cs_top10 = df_top10.groupby(["calligrapher", "script"]).size().reset_index(name="count")
    cs_top10 = cs_top10.sort_values(by=["calligrapher", "count"], ascending=[True, False])
    
    print(f"\n【2. Top 10 书家的 (书家 x 书体) 细分槽位分布 (共 {len(cs_top10)} 个槽位)】:")
    for c in top10_cals:
        sub = cs_top10[cs_top10["calligrapher"] == c]
        slots = [f"{r['script']}:{r['count']}" for _, r in sub.iterrows()]
        print(f"  - {c:<8}: {', '.join(slots)}")

    # 3. 审计当前的 eval 集
    print("\n【3. 当前 eval 评测集与 Top 10 兼容性审计】:")
    print(f"当前 strict 集样本数: {len(df_eval_strict)} 行")
    strict_cals = df_eval_strict["calligrapher"].value_counts()
    print(f"strict 集中包含的书家总数: {len(strict_cals)} 位")
    
    strict_in_top10 = df_eval_strict[df_eval_strict["calligrapher"].isin(top10_cals)]
    print(f"  - 其中属于 Top 10 的样本数: {len(strict_in_top10)} 行 ({len(strict_in_top10)/len(df_eval_strict)*100:.1f}%)")
    print(f"  - 属于非 Top 10 (其余35位书家) 的样本数: {len(df_eval_strict) - len(strict_in_top10)} 行")

    print("\nstrict 集中 Top 10 各书家的样本分布:")
    print(strict_in_top10["calligrapher"].value_counts().to_string())

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
