#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/merge_other_calligraphers_stats.py — 联合汇总 MCCD 与 HCSU 中所有非 45 书家的完整资源清单"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HCSU_CSV = "assets/hcsu_other_calligraphers.csv"
MCCD_DIR = r"G:\GitHub\DiT\MCCD\MCCD\MCCD-Calligrapher\calligrapher_dataset"
TRAIN_50K = r"G:\GitHub\DiT\assets\train_50k_v2_fixed.csv"

# 从之前的 scan_mccd 提取数据
from aggregate_other_calligraphers import scan_mccd, NAME_MAP

def main():
    df_50k = pd.read_csv(TRAIN_50K)
    c45 = set(df_50k["calligrapher"].unique())

    mccd_stats = scan_mccd()
    df_hcsu = pd.read_csv(HCSU_CSV).set_index("calligrapher")

    all_names = sorted(set(mccd_stats.keys()) | set(df_hcsu.index))

    merged = []
    for name in all_names:
        m = mccd_stats.get(name, {"楷": 0, "行": 0, "隶": 0, "草": 0, "篆": 0, "六体": 0, "chars": set(), "total": 0})
        h = df_hcsu.loc[name].to_dict() if name in df_hcsu.index else {"楷": 0, "行": 0, "隶": 0, "草": 0, "篆": 0, "unique_chars": 0, "total_imgs": 0, "kxl_imgs": 0}

        # 估算字数与图片数
        total_kxl = (m["楷"] + m["行"] + m["隶"]) + (h.get("kai", 0) + h.get("xing", 0) + h.get("li", 0))
        total_imgs = m["total"] + h.get("total_imgs", 0)
        
        kai = m["楷"] + h.get("kai", 0)
        xing = m["行"] + h.get("xing", 0)
        li = m["隶"] + h.get("li", 0)
        cao = m["草"] + h.get("cao", 0)
        zhuan = m["篆"] + h.get("zhuan", 0)

        # 估算唯一字符数
        approx_chars = max(len(m["chars"]), int(h.get("unique_chars", 0)))
        if len(m["chars"]) > 0 and int(h.get("unique_chars", 0)) > 0:
            approx_chars = int(len(m["chars"]) + h.get("unique_chars", 0) * 0.7)  # 考虑重合

        merged.append({
            "calligrapher": name,
            "total_kxl": total_kxl,
            "approx_chars": approx_chars,
            "kai": kai,
            "xing": xing,
            "li": li,
            "cao": cao,
            "zhuan": zhuan,
            "total_imgs": total_imgs,
            "in_hcsu": name in df_hcsu.index,
            "in_mccd": name in mccd_stats
        })

    df_all = pd.DataFrame(merged).sort_values(by="total_kxl", ascending=False).reset_index(drop=True)
    out_csv = "assets/all_other_calligraphers_stats.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"全量非 45 书家汇总表已保存至: {out_csv} (共 {len(df_all)} 位)")

    print("\n" + "=" * 75)
    print("HCSU + MCCD 联合汇总：【楷/行/隶】可用资源最丰富的 Top 30 其他名家全景")
    print("=" * 75)
    top30 = df_all.head(30)
    print(f"{'排名':<4} | {'书法家':<8} | {'楷行隶总量':>9} | {'字数(约)':>7} | {'楷':>5} | {'行':>5} | {'隶':>5} | {'草':>5} | {'篆':>5} | {'来源'}")
    print("-" * 75)
    for idx, r in top30.iterrows():
        src = []
        if r['in_mccd']: src.append('MCCD')
        if r['in_hcsu']: src.append('HCSU')
        src_str = '+'.join(src)
        print(f"{idx+1:<4} | {r['calligrapher']:<8} | {r['total_kxl']:>9} | {r['approx_chars']:>7} | {r['kai']:>5} | {r['xing']:>5} | {r['li']:>5} | {r['cao']:>5} | {r['zhuan']:>5} | {src_str}")
    print("-" * 75)

if __name__ == "__main__":
    main()
