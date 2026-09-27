#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/build_top10_eval_sets.py — 构建 Top 10 专属的评测集 (对照集 84 行 + Seen 20 行)"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

EVAL_STRICT_FIXED = "assets/eval_v13_strict_fixed.csv"
TRAIN_TOP10 = "assets/train_top10_manifest.csv"
MAP_JSON = "assets/callig_script_id_map_top10.json"

TOP10_CALS = ["王羲之", "苏轼", "赵孟頫", "欧阳询", "颜真卿", "褚遂良", "米芾", "柳公权", "何绍基", "文徵明"]

def main():
    slot_map = json.load(open(MAP_JSON, encoding="utf-8"))["slot_map"] if os.path.exists(MAP_JSON) else {}

    # 1. 抽取历史对照集 84 行
    df_strict = pd.read_csv(EVAL_STRICT_FIXED)
    df_strict_top10 = df_strict[df_strict["calligrapher"].isin(TOP10_CALS)].copy().reset_index(drop=True)
    df_strict_top10["slot_name"] = df_strict_top10.apply(lambda r: f"{r['calligrapher']}_{r['script']}", axis=1)
    df_strict_top10["slot_id"] = df_strict_top10["slot_name"].map(slot_map)

    out_strict_84 = "assets/eval_top10_strict_subset84.csv"
    df_strict_top10.to_csv(out_strict_84, index=False, encoding="utf-8")
    print(f"🎉 历史对照评估集已生成: {out_strict_84} ({len(df_strict_top10)} 行)")

    # 2. 从 Top 10 训练集中抽取 20 行 Seen 评估集 (每个书家抽 2 行)
    df_train = pd.read_csv(TRAIN_TOP10)
    seen_rows = []
    for c in TOP10_CALS:
        sub = df_train[df_train["calligrapher"] == c]
        # 均匀抽 2 张
        if len(sub) >= 2:
            picked = sub.sample(2, random_state=42)
            seen_rows.append(picked)

    df_seen = pd.concat(seen_rows, ignore_index=True)
    df_seen["slot_name"] = df_seen.apply(lambda r: f"{r['calligrapher']}_{r['script']}", axis=1)
    df_seen["slot_id"] = df_seen["slot_name"].map(slot_map)

    out_seen = "assets/eval_top10_seen_20.csv"
    df_seen.to_csv(out_seen, index=False, encoding="utf-8")
    print(f"🎉 Top 10 Seen 重构集已生成: {out_seen} ({len(df_seen)} 行)")

if __name__ == "__main__":
    import json
    main()
