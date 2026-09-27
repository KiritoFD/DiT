#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/scan_remote_hcsu.py — 审计远程 HCSU 中属于 45 书家未入库的样本"""
import os
import sys
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

df_50k = pd.read_csv("/root/Workspace/xy/DiT/assets/train_50k_v2_fixed.csv")
c45 = set(df_50k["calligrapher"].unique())

# 已进入 50k 的 hcsu_wild 样本 (用字符+书体+书家+图片名比对)
in_50k_hcsu_chars = set()
for _, r in df_50k[df_50k["source"] == "hcsu_wild"].iterrows():
    p = str(r["src_image_path"])
    # 形如: data/hcsu_kxl/imgs/wild/于右任-楷/㑺.png
    parts = p.split("/")
    if len(parts) >= 5:
        dir_name = parts[4]
        fn = parts[-1]
        in_50k_hcsu_chars.add((dir_name, fn))

root = "/root/Workspace/xy/HCSU/wild_extract"
dirs = sorted(os.listdir(root))

alias_map = {"文征明": "文徵明", "赵孟𫖯": "赵孟頫"}

c45_dirs = []
for d in dirs:
    parts = d.split("-")
    if len(parts) == 2:
        c_raw, s = parts
        c = alias_map.get(c_raw, c_raw)
        if c in c45 and s in ["楷", "行", "隶"]:
            c45_dirs.append((d, c, s))

print(f"HCSU 中属于 45 书家且为【楷/行/隶】的目录数: {len(c45_dirs)} 个")

salvage_cands = []
for d, c, s in c45_dirs:
    dp = os.path.join(root, d)
    files = [f for f in os.listdir(dp) if f.lower().endswith(".png")]
    for fn in files:
        if (d, fn) not in in_50k_hcsu_chars:
            ch = os.path.splitext(fn)[0]
            salvage_cands.append({
                "path": os.path.join(dp, fn),
                "calligrapher": c,
                "script": s,
                "character": ch,
                "filename": fn
            })

df_hcsu_cand = pd.DataFrame(salvage_cands)
print(f"➡ HCSU 中属于 45 书家且【未被 50k 采纳】的待打捞样本: {len(df_hcsu_cand):,} 张")

if len(df_hcsu_cand) > 0:
    print("\n待打捞书体分布:")
    print(df_hcsu_cand["script"].value_counts())
    print("\n待打捞书家分布 Top 10:")
    print(df_hcsu_cand["calligrapher"].value_counts().head(10))

    # 抽样检测可用性 (能否打开、是否损坏)
    sample_pool = df_hcsu_cand.sample(min(1000, len(df_hcsu_cand)), random_state=42)
    valid_count = 0
    for idx, row in sample_pool.iterrows():
        try:
            im = Image.open(row["path"])
            if im.size[0] >= 32 and im.size[1] >= 32:
                valid_count += 1
        except Exception:
            pass
    print(f"\n抽检 {len(sample_pool)} 张可用性通过率: {valid_count/len(sample_pool)*100:.1f}%")
