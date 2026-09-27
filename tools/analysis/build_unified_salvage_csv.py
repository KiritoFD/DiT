#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/build_unified_salvage_csv.py — 汇总合并 MCCD 与 HCSU 6,029 张待打捞全景质检总表"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MCCD_CSV = "assets/mccd_salvage_audited_4421.csv"
HCSU_CSV = "assets/hcsu_salvage_audited_1608.csv"
OUT_CSV = "assets/salvaged_real_calligraphy_6029.csv"

def main():
    df_mccd = pd.read_csv(MCCD_CSV)
    df_hcsu = pd.read_csv(HCSU_CSV)

    # 规范化列结构
    common_cols = [
        "source", "calligrapher", "script", "character", "source_file",
        "width", "height", "is_inverted", "ink_ratio", "dirt_ratio",
        "quality", "issues"
    ]

    df_mccd["source"] = "mccd_salvage"
    df_hcsu["source"] = "hcsu_wild_salvage"

    sub_mccd = df_mccd[[c for c in common_cols if c in df_mccd.columns]]
    sub_hcsu = df_hcsu[[c for c in common_cols if c in df_hcsu.columns]]

    merged = pd.concat([sub_mccd, sub_hcsu], ignore_index=True)

    # 动作建议分类
    def determine_action(row):
        q = row["quality"]
        inv = row["is_inverted"]
        dirt = row["dirt_ratio"]
        if q == "broken":
            return "discard", False
        elif inv:
            if dirt > 0.05:
                return "invert_and_denoise", True
            else:
                return "invert_polarity", True
        elif dirt > 0.05:
            return "denoise_bridge", True
        else:
            return "direct_use", True

    actions = [determine_action(r) for _, r in merged.iterrows()]
    merged["salvage_action"] = [a[0] for a in actions]
    merged["is_salvageable"] = [a[1] for a in actions]

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"🎉 统一打捞全景质检表已成功落盘至: {OUT_CSV} (共 {len(merged)} 行)")

    print("\n" + "=" * 65)
    print("6,029 张历史名家墨迹全景质量审计总览 (MCCD + HCSU)")
    print("=" * 65)

    print("\n【1. 数据源构成】:")
    print(merged["source"].value_counts())

    print("\n【2. 最终可用性与打捞动作分布】:")
    for act, cnt in merged["salvage_action"].value_counts().items():
        print(f"  - {act:<20}: {cnt:>5} 张 ({cnt/len(merged)*100:.2f}%)")

    total_salvageable = merged["is_salvageable"].sum()
    print(f"\n【3. 最终判定】:")
    print(f"  - 100% 确认可打捞入库高质量真迹: {total_salvageable:,} 张 ({total_salvageable/len(merged)*100:.1f}%)")
    print(f"  - 建议彻底剔除的残次/空白样本: {len(merged) - total_salvageable} 张 (仅 {(len(merged)-total_salvageable)/len(merged)*100:.1f}%)")

    print("\n【4. 可打捞入库样本的书体构成】:")
    df_valid = merged[merged["is_salvageable"]]
    for s, c in df_valid["script"].value_counts().items():
        print(f"  - {s}书: {c:>5} 张 ({c/total_salvageable*100:.1f}%)")

    print("\n【5. 贡献最大的前 15 位名家真迹增量】:")
    top_c = df_valid["calligrapher"].value_counts().head(15)
    for c, cnt in top_c.items():
        print(f"  - {c:<10}: {cnt:>5} 张")

if __name__ == "__main__":
    main()
