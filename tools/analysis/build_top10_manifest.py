#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/build_top10_manifest.py — 筛选 Top 10 书家的 23 槽位样本清单并构建映射"""
import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_72K = "assets/train_50k_v2_augmented_glyph15k_ext.csv"
OUT_CSV = "assets/train_top10_manifest.csv"
MAP_JSON = "assets/callig_script_id_map_top10.json"

TOP10_CALS = ["王羲之", "苏轼", "赵孟頫", "欧阳询", "颜真卿", "褚遂良", "米芾", "柳公权", "何绍基", "文徵明"]

# 允许的 23 个核心槽位 (过滤掉 <30 的欧阳询隶4、颜真卿隶8、米芾隶2)
ALLOWED_SLOTS = {
    ("王羲之", "行"), ("王羲之", "楷"),
    ("苏轼", "行"), ("苏轼", "楷"),
    ("赵孟頫", "行"), ("赵孟頫", "楷"), ("赵孟頫", "隶"),
    ("欧阳询", "楷"), ("欧阳询", "行"),
    ("颜真卿", "楷"), ("颜真卿", "行"),
    ("褚遂良", "楷"), ("褚遂良", "行"),
    ("米芾", "行"), ("米芾", "楷"),
    ("柳公权", "楷"), ("柳公权", "行"),
    ("何绍基", "隶"), ("何绍基", "行"), ("何绍基", "楷"),
    ("文徵明", "行"), ("文徵明", "楷"), ("文徵明", "隶")
}

def main():
    df = pd.read_csv(TRAIN_72K)
    print(f"输入全量 72k 数据集: {len(df):,} 行")

    # 过滤出 23 槽位样本
    mask = df.apply(lambda r: (r["calligrapher"], r["script"]) in ALLOWED_SLOTS, axis=1)
    df_top10 = df[mask].copy().reset_index(drop=True)
    print(f"Top 10 (23 黄金槽位) 样本总量: {len(df_top10):,} 张")

    # 构建 23 个 slot 的 id 映射
    # slot 命名形式: "书家_书体"，如 "王羲之_行"
    sorted_slots = sorted(list(ALLOWED_SLOTS))
    slot_id_map = {}
    pair_id = 0
    for c, s in sorted_slots:
        slot_key = f"{c}_{s}"
        slot_id_map[slot_key] = pair_id
        pair_id += 1

    print(f"\n构建完成 {len(slot_id_map)} 个槽位映射:")
    for k, v in slot_id_map.items():
        sub_cnt = sum((df_top10["calligrapher"] == k.split("_")[0]) & (df_top10["script"] == k.split("_")[1]))
        print(f"  slot {v:>2}: {k:<10} -> {sub_cnt:>5} 张")

    with open(MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({"num_slots": len(slot_id_map), "slot_map": slot_id_map}, f, ensure_ascii=False, indent=2)
    print(f"\n槽位映射表已保存至: {MAP_JSON}")

    # 给 df_top10 分配 pair_id
    df_top10["slot_name"] = df_top10.apply(lambda r: f"{r['calligrapher']}_{r['script']}", axis=1)
    df_top10["pair_id"] = df_top10["slot_name"].map(slot_id_map)

    # 导出临时清单
    df_top10.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Top 10 清单已导出至: {OUT_CSV} ({len(df_top10)} 行)")

if __name__ == "__main__":
    main()
