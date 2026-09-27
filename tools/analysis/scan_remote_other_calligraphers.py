#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/scan_remote_other_calligraphers.py — 汇总 HCSU (wild/tie/bei) 中所有非 45 类的其他书法家资源"""
import os
import sys
import json
from collections import defaultdict, Counter
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HCSU_ROOT = "/root/Workspace/xy/HCSU"
TRAIN_CSV = "/root/Workspace/xy/DiT/assets/train_50k_v2_fixed.csv"

df_50k = pd.read_csv(TRAIN_CSV)
c45 = set(df_50k["calligrapher"].unique())

NAME_MAP = {
    "赵孟𫖯": "赵孟頫",
    "赵孟頫": "赵孟頫",
    "文征明": "文徵明",
    "文徵明": "文徵明",
    "郑燮": "郑板桥",
    "郑板桥": "郑板桥",
    "赵构": "宋高宗",
    "宋高宗": "宋高宗",
    "朱耷": "八大山人",
    "八大山人": "八大山人",
    "李隆基": "唐玄宗",
    "唐玄宗": "唐玄宗",
    "赵佶": "宋徽宗",
    "宋徽宗": "宋徽宗",
}

def scan_hcsu():
    sources = ["wild_extract", "tie_extract", "bei_extract"]
    stats = defaultdict(lambda: {"楷": 0, "行": 0, "隶": 0, "草": 0, "篆": 0, "chars": set(), "total": 0})

    for s_name in sources:
        sp = os.path.join(HCSU_ROOT, s_name)
        if not os.path.isdir(sp):
            continue
        for d in os.listdir(sp):
            dp = os.path.join(sp, d)
            if not os.path.isdir(dp):
                continue
            parts = d.split("-")
            if len(parts) < 2:
                continue
            c_raw, sc = parts[0], parts[1]
            c_norm = NAME_MAP.get(c_raw, c_raw)
            if c_norm in c45 or c_raw in c45:
                continue

            files = [f for f in os.listdir(dp) if f.lower().endswith((".png", ".jpg"))]
            for fn in files:
                ch = os.path.splitext(fn)[0]
                stats[c_norm]["total"] += 1
                stats[c_norm]["chars"].add(ch)
                if sc in stats[c_norm]:
                    stats[c_norm][sc] += 1
                else:
                    stats[c_norm]["其他"] = stats[c_norm].get("其他", 0) + 1

    return stats

def main():
    hcsu_stats = scan_hcsu()
    print(f"HCSU 中非 45 书家的独立名家数: {len(hcsu_stats)} 位")

    records = []
    for c, d in hcsu_stats.items():
        kxl = d["楷"] + d["行"] + d["隶"]
        records.append({
            "calligrapher": c,
            "total_imgs": d["total"],
            "unique_chars": len(d["chars"]),
            "kxl_imgs": kxl,
            "kai": d["楷"],
            "xing": d["行"],
            "li": d["隶"],
            "cao": d["草"],
            "zhuan": d["篆"]
        })
    df = pd.DataFrame(records).sort_values(by="kxl_imgs", ascending=False).reset_index(drop=True)
    out_csv = "/root/Workspace/xy/DiT/assets/hcsu_other_calligraphers.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"已导出至: {out_csv}")

    print("\nHCSU 中楷行隶资源最丰富的其他名家 Top 20:")
    print(df[["calligrapher", "kxl_imgs", "unique_chars", "kai", "xing", "li", "total_imgs"]].head(20).to_string())

if __name__ == "__main__":
    main()
