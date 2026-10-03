#!/usr/bin/env python
"""把表里 cos≈1 的塌缩行对的实际图片拼出来, 肉眼验证是否同一个字。

同时校验 npz glyph_ids 的口径:
  假说A: glyph_id = csv 的 glyph_id 列 (全局字 id)
  假说B: glyph_id = script_id*7026 + character_id (dit.py 口径)
用 npz 里已有的 (calligs, scripts) 对照 csv, 看哪个假说成立。
"""
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

os.chdir("/root/Workspace/xy/DiT")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CH = 7026

# ── csv 索引 ──
rows = []
by_img = {}
for r in csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8")):
    rows.append(r)
    by_img[int(r["img_id"])] = r
print(f"[csv] {len(rows)} 行")

# ── npz glyph 口径校验 ──
z = np.load("assets/dino_feat_top10_g.npz")
g_np = z["glyph_ids"].astype(np.int64)
c_np = z["calligs"].astype(np.int64)
s_np = z["scripts"].astype(np.int64)
hypA = hypB = 0
for i in range(len(g_np)):
    r = by_img.get(i)
    if r is None:
        continue
    if int(r["glyph_id"]) == int(g_np[i]):
        hypA += 1
    if int(r["script_id"]) * CH + int(r["character_id"]) == int(g_np[i]):
        hypB += 1
print(f"[口径] 假说A (glyph_id=csv全局字id): {hypA} 命中 | "
      f"假说B (script*7026+char): {hypB} 命中 / 共 {len(g_np)}")

# ── 表 + 塌缩对 ──
idx = json.load(open("assets/char_script_supcon_index.json", encoding="utf-8"))
glyphs = idx["glyphs"]
W = torch.from_numpy(np.load("assets/char_script_supcon_table.npy")).float()[glyphs].to(DEV)
Wn = F.normalize(W, dim=-1)
S = Wn @ Wn.T
S.fill_diagonal_(-1)
iu = torch.triu_indices(S.shape[0], S.shape[0], 1, device=DEV)
pair_cos = S[iu[0], iu[1]]
order = pair_cos.argsort(descending=True)

# glyph -> csv 行 (按假说A查; 若假说B成立再用B)
def rows_for_glyph(g):
    out = []
    for r in rows:
        if int(r["glyph_id"]) == g:          # 假说A口径
            out.append(r)
    return out[:3]

def rows_for_glyph_B(g):
    si, ci = g // CH, g % CH
    out = []
    for r in rows:
        if int(r["script_id"]) == si and int(r["character_id"]) == ci:
            out.append(r)
    return out[:3]

picker = rows_for_glyph if hypA >= hypB else rows_for_glyph_B
print(f"[picker] 用假说{'A' if hypA >= hypB else 'B'} 找图")

# ── 拼图: 每对一行, 左右各最多 3 张 ──
CELL, LBL = 128, 22
top = order[:12].tolist()
pairs_meta = []
imgs_all = []
for t in top:
    i, j = int(iu[0][t]), int(iu[1][t])
    gi, gj = glyphs[i], glyphs[j]
    c_tab = float(pair_cos[t])
    ri, rj = picker(gi), picker(gj)
    pairs_meta.append((gi, gj, c_tab, ri, rj))

n_rows = len(pairs_meta)
max_cols = max(max(len(a[3]), len(a[4])) for a in pairs_meta)
W_img = max_cols * 2 * CELL + 3 * CELL + 40   # 左组 | 右组 | 中间留白
H_img = n_rows * (CELL + LBL) + 10
canvas = Image.new("RGB", (W_img, H_img), (24, 24, 24))
dr = ImageDraw.Draw(canvas)

def paste(r, x, y, tag):
    if not r:
        return
    p = r["image_path"]
    if not os.path.exists(p):
        dr.text((x, y + CELL // 2), f"缺图 {p}", fill=(255, 80, 80))
        return
    im = Image.open(p).convert("RGB").resize((CELL, CELL))
    canvas.paste(im, (x, y))
    dr.text((x, y + CELL + 2), f"{r['character']} {r['calligrapher']}{r['script']} #{r['img_id']}",
            fill=(220, 220, 120))
    if tag:
        dr.text((x, y - 16), tag, fill=(120, 200, 255))

for k, (gi, gj, c_tab, ri, rj) in enumerate(pairs_meta):
    y = k * (CELL + LBL) + 20
    for m, r in enumerate(ri):
        paste(r, m * CELL, y, f"glyph {gi}" if m == 0 else "")
    xm = max_cols * CELL + 10
    dr.text((xm, y + CELL // 2), f"cos\n{c_tab:.4f}", fill=(255, 255, 255))
    for m, r in enumerate(rj):
        paste(r, (max_cols + 2 + m) * CELL, y, f"glyph {gj}" if m == 0 else "")

out = "exp-std/collapse_pairs_montage.png"
canvas.save(out)
print(f"[ok] -> {out}  ({W_img}x{H_img}, {n_rows} 对)")
for gi, gj, c_tab, ri, rj in pairs_meta[:12]:
    ci = ri[0]["character"] if ri else "?"
    cj = rj[0]["character"] if rj else "?"
    print(f"  {c_tab:.4f}  glyph {gi}='{ci}' vs {gj}='{cj}'  "
          f"(A口径行数 {len(ri)}/{len(rj)})")
