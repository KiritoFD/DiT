# -*- coding: utf-8 -*-
"""构建「标准字形 DINO」char embedding 表。

对齐方式
--------
当前模型 (factorized_add) char 条件按 char_id (0..7025) 组织，glyph_id = script_id*7026 + char_id。
标准字形 DINO 特征是 codepoint 级（std_{kai,li}_cls.npy + index codepoint）。
通过 charid2char -> codepoint 映射，建 char_id 级 embedding 表。

策略
----
- 优先 kai，缺则用 li，都缺用「全局均值 + 扰动」作为 fallback（可学习？这里冻结）。
- 输出: char_embed_table.npy (7026, 768)，冻结查表。可训练参数 = 0。
- 也存一份 384d 版本？DINO 标准字形没有原生 384，先用 768。若模型 char_embed_dim=384，
  需投影（可训练 ~590K<2M），或改配置。先输出 768。
"""
import os, sys, csv, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
NUM_CH = 7026

# char_id -> character
charid2char = {}
for r in csv.DictReader(open("assets/train_fame.csv", encoding="utf-8")):
    charid2char[int(r["character_id"])] = r["character"]

# 加载标准字形特征
kai_cls = np.load("_sync_work/std_kai_cls.npy").astype(np.float32)
kai_cp = json.load(open("_sync_work/std_kai_index.json", encoding="utf-8"))["codepoints"]
li_cls = np.load("_sync_work/std_li_cls.npy").astype(np.float32)
li_cp = json.load(open("_sync_work/std_li_index.json", encoding="utf-8"))["codepoints"]
kai_map = {cp: e for cp, e in zip(kai_cp, kai_cls)}
li_map = {cp: e for cp, e in zip(li_cp, li_cls)}

# 归一化 helper
def norm(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n

D = kai_cls.shape[1]
table = np.zeros((NUM_CH, D), dtype=np.float32)
cover_src = {}
covered = 0
for cid in range(NUM_CH):
    ch = charid2char.get(cid)
    if not ch or len(ch) != 1:
        cover_src[cid] = "no_char"
        continue
    cp = ord(ch)
    if cp in kai_map:
        table[cid] = norm(kai_map[cp]); cover_src[cid] = "kai"; covered += 1
    elif cp in li_map:
        table[cid] = norm(li_map[cp]); cover_src[cid] = "li"; covered += 1
    else:
        cover_src[cid] = "missing"

# fallback: 已覆盖的全局均值
ok_rows = np.array([cover_src.get(c) in ("kai", "li") for c in range(NUM_CH)])
fallback = norm(table[ok_rows].mean(axis=0)) if ok_rows.any() else np.zeros(D)
n_missing = sum(1 for c in range(NUM_CH) if cover_src.get(c) == "missing")
for cid in range(NUM_CH):
    if cover_src.get(cid) == "missing":
        table[cid] = fallback

np.save("_sync_work/std_dino_char_table_768.npy", table)
# 统计
from collections import Counter
cnt = Counter(cover_src.values())
print(f"char 表: {table.shape}  D={D}")
print(f"覆盖: kai={cnt['kai']} li={cnt['li']} missing={cnt['missing']} no_char={cnt.get('no_char',0)}")
print(f"训练集出现字符覆盖数: {covered}/{len([c for c in range(NUM_CH) if charid2char.get(c)])}")
print(f"已保存 _sync_work/std_dino_char_table_768.npy")

# 快速验证：形近对在表里的余弦
SIM = [("土","士"),("大","太"),("王","玉"),("日","曰"),("未","末"),("人","入")]
def cid_of(ch):
    for cid, cc in charid2char.items():
        if cc == ch:
            return cid
    return None
print("\n形近对 char 表余弦 (应为高):")
for a, b in SIM:
    ca, cb = cid_of(a), cid_of(b)
    if ca is not None and cb is not None:
        print(f"  {a}/{b}: cos={float(table[ca] @ table[cb]):.3f}")
