#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/inspect_mccd_salvage.py — 全面盘点 MCCD 原始数据中的可用量与遗失量"""
import os
import sys
import json
from collections import defaultdict, Counter
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MCCD_DIR = r"G:\GitHub\DiT\MCCD\MCCD\MCCD-Calligrapher\calligrapher_dataset"
TRAIN_50K = r"G:\GitHub\DiT\assets\train_50k_v2_fixed.csv"

def main():
    print("=" * 70)
    print("MCCD 原始数据集盘点与未利用/遗失样本深度审计")
    print("=" * 70)

    df_50k = pd.read_csv(TRAIN_50K)
    c45_counts = dict(df_50k["calligrapher"].value_counts())
    c45_set = set(c45_counts.keys())
    print(f"当前 50k 底库中包含: {len(c45_set)} 位书家, 总计 {len(df_50k)} 条样本")

    all_callig_dirs = sorted(os.listdir(MCCD_DIR))
    print(f"MCCD-Calligrapher 包含目录数: {len(all_callig_dirs)} 个")

    alias_map = {
        "\u8d75\u5b5f\U0002b5af": "赵孟頫",
        "文征明": "文徵明",
    }

    # 统计 MCCD 全局总览
    total_imgs = 0
    script_counts = Counter()
    callig_imgs = {}
    callig_kxl_imgs = defaultdict(int)

    for d in all_callig_dirs:
        dp = os.path.join(MCCD_DIR, d)
        if not os.path.isdir(dp):
            continue
        c_norm = alias_map.get(d, d)
        files = [f for f in os.listdir(dp) if f.lower().endswith((".png", ".jpg"))]
        callig_imgs[c_norm] = len(files)
        total_imgs += len(files)

        for fn in files:
            parts = fn.rsplit(".", 1)[0].split("-")
            if len(parts) >= 2:
                s = parts[1]
                script_counts[s] += 1
                if s in ["楷", "行", "隶"]:
                    callig_kxl_imgs[c_norm] += 1

    print(f"\nMCCD 原始图片全集总数: {total_imgs:,} 张")
    print("\nMCCD 书体全景分布:")
    for s, c in script_counts.most_common():
        print(f"  - {s:<8}: {c:>6,} 张 ({c/total_imgs*100:.1f}%)")

    # 分类：45 类已有书家 vs 100+ 类未收录书家
    mccd_in_45 = set(callig_imgs.keys()) & c45_set
    mccd_not_in_45 = set(callig_imgs.keys()) - c45_set

    imgs_in_45_all = sum(callig_imgs[c] for c in mccd_in_45)
    imgs_in_45_kxl = sum(callig_kxl_imgs[c] for c in mccd_in_45)

    imgs_not_in_45_all = sum(callig_imgs[c] for c in mccd_not_in_45)
    imgs_not_in_45_kxl = sum(callig_kxl_imgs[c] for c in mccd_not_in_45)

    print(f"\n【维度一：45 类已收录名家在 MCCD 中的存量情况】")
    print(f"  - 45 位书家中在 MCCD 有对应目录的有: {len(mccd_in_45)} 位")
    print(f"  - 这 {len(mccd_in_45)} 位书家在 MCCD 的图片总量: {imgs_in_45_all:,} 张")
    print(f"  - 这 {len(mccd_in_45)} 位书家在 MCCD 的【楷/行/隶】真迹总量: {imgs_in_45_kxl:,} 张")
    print(f"  - 当前 50k 底库中来自原名家底库 (final_imgs_fame_v8) 的仅有: 26,171 张")
    print(f"  - ➡ 即使只看这 37 位已有书家，在 MCCD 的楷行隶中仍有至少 {imgs_in_45_kxl - 26171:,} 张真实真迹未进入底库！")

    print("\n【维度二：未收录的 100+ 位古代/近代名家（全集）】")
    print(f"  - MCCD 中完全未被我们收录的书家: {len(mccd_not_in_45)} 位")
    print(f"  - 未收录书家图片总量: {imgs_not_in_45_all:,} 张")
    print(f"  - 未收录书家中的【楷/行/隶】总量: {imgs_not_in_45_kxl:,} 张！")

    print("\n未收录书家中【楷/行/隶】资源最丰富的 Top 15 大师:")
    top_unused = sorted([(c, callig_kxl_imgs[c], callig_imgs[c]) for c in mccd_not_in_45], key=lambda x: -x[1])
    for c, kxl_cnt, all_cnt in top_unused[:15]:
        print(f"  - {c:<10}: 楷行隶 {kxl_cnt:>4} 张 (总存量 {all_cnt:>4} 张)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
