# -*- coding: utf-8 -*-
"""
build_base_noaug.py — fame-tj base 数据集 (无增强): fame3 原始 + tongji 原始.

  fame: train_fame3_clean_v8.csv (28,385 原始行, 无增强) — 楷0/行3/隶4
  + tongji 原始 2,992 行 (无增强, 同 build_fame_tj_kxl 的过滤: 排除草书/strict 泄漏,
    新字 50000+ 段, 新书家 raw 9000+)
  (+ UniCalli 拿到授权后追加)

输出: assets/train_base_noaug.csv (列与 fame 一致)
      assets/callig_id_map_base.json
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAME_CSV = "assets/train_fame3_clean_v8.csv"
OUT_CSV = "assets/train_base_noaug.csv"
OUT_MAP = "assets/callig_id_map_base.json"
KEEP_SCRIPTS = {"楷", "行", "隶"}

# ── fame 字表 + 原始行 ─────────────────────────────────────────────────────
char_map = {}
fame_rows = []
with open(FAME_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        char_map.setdefault(r["character"], (r["character_id"], r["glyph_id"]))
        if r["script"] in KEEP_SCRIPTS:
            row = dict(r)
            row.setdefault("aug", "")
            fame_rows.append(row)
print(f"fame base rows: {len(fame_rows)}")

raw_ids = set()
callig_name2raw = {}
with open(FAME_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        raw_ids.add(int(r["calligrapher_id"]))
        callig_name2raw[r["calligrapher"]] = int(r["calligrapher_id"])

strict_pairs = set()
with open("assets/eval_fame3_strict_clean_v9.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        strict_pairs.add((r["calligrapher"], r["script"], r["character"]))

# ── tongji 原始行 (与 fame-tj-kxl 同过滤, 无增强) ─────────────────────────
import glob
import shutil
TONGJI_IMG_ROOT = "data/imgs/calli_tongji"
TONGJI_FLAT = "data/imgs/calli_tongji_imgs"
tongji_rows = []
new_calligs = {}
skipped = {"grass": 0, "leak": 0}
iid = 950000
for d in sorted(glob.glob(os.path.join(TONGJI_IMG_ROOT, "*"))):
    name = os.path.basename(d)
    callig, script = name.rsplit("-", 1)
    if script not in KEEP_SCRIPTS:
        skipped["grass"] += len(glob.glob(os.path.join(d, "*.png")))
        continue
    if callig not in callig_name2raw:
        callig_name2raw[callig] = 9000 + len(new_calligs)
        new_calligs[callig] = 9000 + len(new_calligs)
    cid = str(callig_name2raw[callig])
    for p in sorted(glob.glob(os.path.join(d, "*.png"))):
        ch = os.path.basename(p)[:-4]
        if (callig, script, ch) in strict_pairs:
            skipped["leak"] += 1
            continue
        if ch not in char_map:
            char_map[ch] = (str(50000 + len(char_map)), str(50000 + len(char_map)))
        csid, gid = char_map[ch]
        dst = os.path.join(TONGJI_FLAT, f"{iid}.png")
        os.makedirs(TONGJI_FLAT, exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(p, dst)
        tongji_rows.append({
            "image_path": f"{TONGJI_FLAT}/{iid}.png", "calligrapher": callig,
            "script": script, "character": ch, "calligrapher_id": cid,
            "script_id": {"楷": "0", "行": "3", "隶": "4"}[script],
            "character_id": csid, "glyph_id": gid, "aug": "",
        })
        iid += 1
print(f"tongji base rows: {len(tongji_rows)}, skipped={skipped}")

fields = ["image_path", "calligrapher", "script", "character", "calligrapher_id",
          "script_id", "character_id", "glyph_id", "aug"]
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in fame_rows:
        w.writerow({k: r[k] for k in fields})
    for r in tongji_rows:
        w.writerow({k: r[k] for k in fields})
print(f"written {OUT_CSV}: {len(fame_rows)} + {len(tongji_rows)} rows")

# ── callig id map ──────────────────────────────────────────────────────────
with open("assets/callig_id_map.json", encoding="utf-8") as f:
    cmap = json.load(f)
id_map = {str(k): int(v) for k, v in cmap.get("id_map", {}).items()}
n_cont = int(cmap.get("num_calligraphers", 41))
for callig, new_raw in sorted(new_calligs.items()):
    id_map[str(new_raw)] = n_cont + sorted(new_calligs).index(callig)
cmap["id_map"] = id_map
cmap["num_calligraphers"] = n_cont + len(new_calligs)
with open(OUT_MAP, "w", encoding="utf-8") as f:
    json.dump(cmap, f, ensure_ascii=False, indent=2)
print(f"written {OUT_MAP}: num_calligraphers={cmap['num_calligraphers']}")
