#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 MCCD 文件名 → (character, script_id, calligrapher_id, glyph_id) 映射"""
import os, csv, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MCCD_DIR = r"G:\GitHub\DiT\MCCD\MCCD\MCCD-Calligrapher\calligrapher_dataset"
CALLIGS_JSON = r"G:\GitHub\DiT\5script\top30_calligs_clean.json"
TRAIN_CSV = r"G:\GitHub\DiT\5script\train_top30_clean.csv"

# script 名 → script_id
SCRIPT_MAP = {"楷": 0, "篆": 1, "草": 2, "行": 3, "隶": 4}

# 读取 calligrapher 名单 (id, name)
with open(CALLIGS_JSON, encoding="utf-8") as f:
    calligs_data = json.load(f)
callig_name2id = {}  # name -> calligrapher_id (全局唯一)
for sid_str, lst in calligs_data.items():
    for c in lst:
        callig_name2id[c["name"]] = int(c["id"])

# 读取 character → character_id 映射 (从 train CSV)
char2id = {}
with open(TRAIN_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        char2id[r["character"]] = int(r["character_id"])

print(f"Calligraphers: {len(callig_name2id)}")
print(f"Characters: {len(char2id)}")

# 遍历 MCCD-Calligrapher 目录
# 文件名格式: 字-书体-朝代-书家-编号.png  例如: 人-楷-唐-吴彩鸾-170027.png
import re
records = []
skipped_script = 0
skipped_callig = 0
skipped_char = 0
total = 0

for callig_dir in sorted(os.listdir(MCCD_DIR)):
    callig_path = os.path.join(MCCD_DIR, callig_dir)
    if not os.path.isdir(callig_path):
        continue
    # callig_dir 就是书家名
    callig_id = callig_name2id.get(callig_dir)
    if callig_id is None:
        skipped_callig += 1
        continue

    for fname in os.listdir(callig_path):
        if not fname.endswith(".png"):
            continue
        total += 1
        # 解析文件名: 字-书体-朝代-书家-编号.png
        parts = fname.rsplit(".", 1)[0].split("-")
        if len(parts) < 5:
            continue
        char = parts[0]
        script_name = parts[1]
        script_id = SCRIPT_MAP.get(script_name)
        if script_id is None:
            skipped_script += 1
            continue
        char_id = char2id.get(char)
        if char_id is None:
            skipped_char += 1
            continue
        # glyph_id = script_id * NUM_CHARACTERS + char_id (和 _add_glyph_col.py 一致)
        glyph_id = script_id * 10000 + char_id  # placeholder, will fix later

        records.append({
            "filepath": os.path.join(callig_path, fname),
            "character": char,
            "script_name": script_name,
            "script_id": script_id,
            "calligrapher": callig_dir,
            "calligrapher_id": callig_id,
            "character_id": char_id,
            "glyph_id": glyph_id,
        })

print(f"\nTotal images scanned: {total}")
print(f"Matched: {len(records)}")
print(f"Skipped (script not in top5): {skipped_script}")
print(f"Skipped (calligrapher not in list): {skipped_callig}")
print(f"Skipped (character not in train set): {skipped_char}")

# 按 script 统计
from collections import Counter
script_counts = Counter(r["script_id"] for r in records)
for sid in range(5):
    sname = {0:"楷",1:"篆",2:"草",3:"行",4:"隶"}[sid]
    print(f"  script {sid} ({sname}): {script_counts[sid]} images")

# 统计 glyph 数
glyphs = set((r["script_id"], r["character_id"]) for r in records)
print(f"\nUnique glyphs (script×char): {len(glyphs)}")

# 写映射 CSV
out = r"G:\GitHub\DiT\5script\mccd_image_map.csv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filepath","character","script_name","script_id",
                                       "calligrapher","calligrapher_id","character_id","glyph_id"])
    w.writeheader()
    w.writerows(records)
print(f"\nMapping CSV → {out} ({len(records)} rows)")
