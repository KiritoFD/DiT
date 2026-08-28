#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""远程验证: clean CSV 的所有 glyph 是否都在 DINO vocab (glyph_dino_index.json) 里。"""
import csv, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
IDX = os.path.join(BASE, "pretrained_models/dino_embeddings/glyph_dino_index.json")
EMB = os.path.join(BASE, "pretrained_models/dino_embeddings/glyph_dino_embeddings.npy")
CSVS = [os.path.join(BASE, "5script/train_top30_clean.csv"),
        os.path.join(BASE, "5script/eval100_top30_clean.csv")]

idx = json.load(open(IDX, encoding="utf-8"))
glyphs = idx["glyphs"] if isinstance(idx, dict) else idx
print(f"DINO vocab: {len(glyphs)} glyphs, dim check:", end=" ")
import numpy as np
emb = np.load(EMB)
print(emb.shape)
assert len(glyphs) == emb.shape[0], "index/emb length mismatch"

vocab = set(tuple(g) for g in glyphs)
NUM_CH = 7026
max_gid = -1
for g in glyphs:
    sid, cid = int(g[0]), int(g[1])
    max_gid = max(max_gid, sid*NUM_CH + cid)
print(f"max glyph_id = {max_gid} (num_characters=35130 -> table 35131 rows, CFG idx 35130) -> {'OK' if max_gid < 35130 else 'OVERFLOW!'}")

for p in CSVS:
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    missing = set()
    for r in rows:
        g = (int(r["script_id"]), int(r["character_id"]))
        if g not in vocab:
            missing.add(g)
    print(f"{os.path.basename(p)}: {len(rows)} rows, {len(missing)} missing glyphs -> {'COVERED' if not missing else missing}")