#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/build_extend_writer_csv.py — 生成规范的 train_extend_writer.csv"""
import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

START_ID = 76000
MANIFEST_5808 = "assets/extend_writer_manifest_5808.csv"
TRAIN_50K = "assets/train_50k_v2_fixed.csv"
MAP_50K = "assets/callig_id_map_50k.json"

def main():
    df_man = pd.read_csv(MANIFEST_5808)
    df_50k = pd.read_csv(TRAIN_50K)

    # 建立书家名 -> calligrapher_id 映射 (使用 50k 底库一致的 id)
    c2id = dict(zip(df_50k["calligrapher"], df_50k["calligrapher_id"]))
    
    # 建立汉字 -> character_id 映射，若生僻字未出现过则顺延分配
    char2id = dict(zip(df_50k["character"], df_50k["character_id"]))
    max_char_id = df_50k["character_id"].max()

    # 建立 (script, char) -> glyph_id 映射
    sc2gid = dict(zip(zip(df_50k["script"], df_50k["character"]), df_50k["glyph_id"]))
    max_glyph_id = df_50k["glyph_id"].max()

    script_ids = {"楷": 0, "行": 3, "隶": 4}

    rows = []
    curr_id = START_ID

    for idx, r in df_man.iterrows():
        c = r["calligrapher"]
        s = r["script"]
        ch = r["character"]

        cid = c2id[c]
        sid = script_ids[s]

        if ch not in char2id:
            max_char_id += 1
            char2id[ch] = max_char_id
        chid = char2id[ch]

        if (s, ch) not in sc2gid:
            max_glyph_id += 1
            sc2gid[(s, ch)] = max_glyph_id
        gid = sc2gid[(s, ch)]

        img_rel = f"data/extend_writer/imgs/{curr_id:06d}.png"
        std_rel = f"data/extend_writer/std/{curr_id:06d}.png"

        rows.append({
            "image_path": img_rel,
            "calligrapher": c,
            "script": s,
            "character": ch,
            "calligrapher_id": cid,
            "script_id": sid,
            "character_id": chid,
            "glyph_id": gid,
            "aug": "",
            "std_path": std_rel,
            "source": r["source"],
            "src_image_path": r["source_file"],
            "is_inverted": r["is_inverted"],
            "extend_writer_id": curr_id
        })
        curr_id += 1

    df_out = pd.DataFrame(rows)
    out_csv = "assets/train_extend_writer.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"🎉 train_extend_writer.csv 已生成至: {out_csv} (共 {len(df_out)} 条记录)")
    print(f"编号范围: {START_ID:06d}.png ~ {curr_id-1:06d}.png")
    print(df_out["source"].value_counts())

if __name__ == "__main__":
    main()
