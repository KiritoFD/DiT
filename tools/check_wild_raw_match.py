# -*- coding: utf-8 -*-
"""check_wild_raw_match.py — 验证 train_50k_v2_fixed_augmented.csv 中的 hcsu_wild 样本与 HCSU/wild_extract 原始图片的一致性。"""
import os
import sys
import pandas as pd

CSV_PATH = "assets/train_50k_v2_fixed_augmented.csv"
WILD_RAW_ROOT = "/root/Workspace/xy/HCSU/wild_extract"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found")
        return
    df = pd.read_csv(CSV_PATH)
    for src_name, raw_root in [
        ("hcsu_wild", "/root/Workspace/xy/HCSU/wild_extract"),
        ("hcsu_bei", "/root/Workspace/xy/HCSU/bei_extract"),
        ("hcsu_tie", "/root/Workspace/xy/HCSU/tie_extract"),
    ]:
        sub_df = df[df["source"] == src_name]
        match_count = 0
        tag = src_name.replace("hcsu_", "")
        prefix = f"data/hcsu_kxl/imgs/{tag}/"
        for _, r in sub_df.iterrows():
            src_p = str(r["src_image_path"])
            rel = src_p.replace(prefix, "")
            raw_p = os.path.join(raw_root, rel)
            if os.path.exists(raw_p):
                match_count += 1
        pct = match_count / len(sub_df) * 100 if len(sub_df) > 0 else 0
        print(f"{src_name}: {match_count}/{len(sub_df)} ({pct:.2f}%)")

if __name__ == "__main__":
    main()
