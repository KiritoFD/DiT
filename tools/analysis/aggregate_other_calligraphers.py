#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/aggregate_other_calligraphers.py — 汇总统计 HCSU 与 MCCD 中所有非 45 类的其他书法家资源"""
import os
import sys
import json
from collections import defaultdict, Counter
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MCCD_DIR = r"G:\GitHub\DiT\MCCD\MCCD\MCCD-Calligrapher\calligrapher_dataset"
HCSU_WILD_DIR = "/root/Workspace/xy/HCSU/wild_extract"  # 在远程或通过列表/本地模拟
TRAIN_50K = r"G:\GitHub\DiT\assets\train_50k_v2_fixed.csv"

# 45 位现有书家名单
df_50k = pd.read_csv(TRAIN_50K)
c45 = set(df_50k["calligrapher"].unique())

# 书家别名与写法归一化表
NAME_MAP = {
    "\u8d75\u5b5f\U0002b5af": "赵孟頫",
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

def scan_mccd():
    """扫描 MCCD 中的其他书家"""
    mccd_stats = defaultdict(lambda: {"楷": 0, "行": 0, "隶": 0, "草": 0, "篆": 0, "六体": 0, "chars": set(), "total": 0})
    for d in os.listdir(MCCD_DIR):
        dp = os.path.join(MCCD_DIR, d)
        if not os.path.isdir(dp):
            continue
        c_norm = NAME_MAP.get(d, d)
        if c_norm in c45 or d in c45:
            continue
        
        for fn in os.listdir(dp):
            if not fn.lower().endswith((".png", ".jpg")):
                continue
            parts = fn.rsplit(".", 1)[0].split("-")
            if len(parts) < 2:
                continue
            ch = parts[0].strip()
            sc = parts[1].strip()
            
            mccd_stats[c_norm]["total"] += 1
            mccd_stats[c_norm]["chars"].add(ch)
            if sc in mccd_stats[c_norm]:
                mccd_stats[c_norm][sc] += 1
            else:
                mccd_stats[c_norm]["其他"] = mccd_stats[c_norm].get("其他", 0) + 1
    return mccd_stats

def main():
    mccd_stats = scan_mccd()
    print(f"MCCD 中非 45 书家的独立大家数: {len(mccd_stats)} 位")

    # 转为 DataFrame
    records = []
    for c, d in mccd_stats.items():
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
            "zhuan": d["篆"],
            "liuti": d["六体"]
        })
    df_mccd = pd.DataFrame(records)
    print("\nMCCD 中楷行隶资源最丰富的 Top 25 位其他名家:")
    df_mccd_kxl = df_mccd.sort_values(by="kxl_imgs", ascending=False).reset_index(drop=True)
    print(df_mccd_kxl[["calligrapher", "kxl_imgs", "unique_chars", "kai", "xing", "li", "total_imgs"]].head(25).to_string())

if __name__ == "__main__":
    main()
