#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/export_clean_salvage_manifest.py — 导出严格排重后的高质量待打捞清单"""
import os
import sys
import json
import re
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MCCD_DIR = r"G:\GitHub\DiT\MCCD\MCCD\MCCD-Calligrapher\calligrapher_dataset"
TRAIN_CSV = r"G:\GitHub\DiT\assets\train_50k_v2_fixed.csv"
MANIFEST = r"G:\GitHub\DiT\archive\final_manifest.json"

def main():
    df_50k = pd.read_csv(TRAIN_CSV)
    fame_manifest_ids = set()
    for p in df_50k[df_50k["source"] == "final_imgs_fame_v8"]["src_image_path"].dropna():
        m = re.search(r"(\d+)\.png$", str(p))
        if m:
            fame_manifest_ids.add(int(m.group(1)))

    with open(MANIFEST, encoding="utf-8") as f:
        manifest_list = json.load(f)

    existing_manifest_seqs = set()
    for item in manifest_list:
        if item["img_id"] in fame_manifest_ids:
            if "orig_seq" in item and item["orig_seq"]:
                existing_manifest_seqs.add(str(item["orig_seq"]))

    c45 = set(df_50k["calligrapher"].unique())
    alias_map = {
        "\u8d75\u5b5f\U0002b5af": "赵孟頫",
        "文征明": "文徵明",
    }

    script_target = {"楷", "行", "隶"}
    clean_records = []

    for orig_d in os.listdir(MCCD_DIR):
        c_norm = alias_map.get(orig_d, orig_d)
        if c_norm not in c45:
            continue
        
        dp = os.path.join(MCCD_DIR, orig_d)
        if not os.path.isdir(dp):
            continue

        for fn in os.listdir(dp):
            if not fn.lower().endswith((".png", ".jpg")):
                continue
            parts = fn.rsplit(".", 1)[0].split("-")
            if len(parts) < 5:
                continue
            ch = parts[0].strip()
            sc = parts[1].strip()
            dy = parts[2].strip()
            seq = parts[4].strip()

            if sc not in script_target:
                continue
            if seq in existing_manifest_seqs:
                continue

            clean_records.append({
                "source_file": os.path.join(dp, fn),
                "calligrapher": c_norm,
                "script": sc,
                "character": ch,
                "dynasty": dy,
                "seq": seq,
                "filename": fn,
                "source": "mccd_salvage"
            })

    df_clean = pd.DataFrame(clean_records).drop_duplicates(subset=["seq"])
    out_p = "assets/mccd_salvage_clean_4421.csv"
    df_clean.to_csv(out_p, index=False, encoding="utf-8")
    print(f"严格排重后的 MCCD 待打捞清单已导出至: {out_p} ({len(df_clean)} 行)")

if __name__ == "__main__":
    main()
