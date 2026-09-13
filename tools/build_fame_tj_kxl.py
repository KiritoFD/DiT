# -*- coding: utf-8 -*-
"""
build_fame_tj_kxl.py — fame-tj-kxl 数据集 v2 (v4 对称增强统一口径).

组成:
  fame: train_fame3_sym_full.csv (v4 对称增强, 楷0/行3/隶4 全部行)
  + tongji 楷/行/隶 2,992 原始 + v4 对称增强变体 (±同幅 dilate/erode, 同 p):
      原图 uid 950000+ (data/imgs/calli_tongji_imgs/)
      tp uid 9800000+ / tn uid 9900000+ (data/imgs/calli_tongji_sym/)
    排除草书 (5 目录 2000 张)、strict eval 泄漏 8 行
    新字 638 个 -> character_id/glyph_id 段 50000+
    新书家 11 个 -> raw id 9000+, 连续 id 41-51 (callig_id_map_tj.json)
输出:
  assets/train_fame_tj_kxl.csv
  assets/callig_id_map_tj.json
"""
import csv
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAME_CSV = "assets/train_fame3_sym_full.csv"
TONGJI_IMG_ROOT = "data/imgs/calli_tongji"
TONGJI_FLAT = "data/imgs/calli_tongji_imgs"
TONGJI_SYM = "data/imgs/calli_tongji_sym"
OUT_CSV = "assets/train_fame_tj_kxl.csv"
OUT_MAP = "assets/callig_id_map_tj.json"
IMG_ID_BASE = 950000
UID_T = 9800000
UID_N = 9900000
KEEP_SCRIPTS = {"楷", "行", "隶"}
ST = generate_binary_structure(2, 2)

# ── fame 字表映射 + sym_full 行 ────────────────────────────────────────────
char_map = {}
fame_rows = []
with open(FAME_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        char_map.setdefault(r["character"], (r["character_id"], r["glyph_id"]))
        if r["script"] in KEEP_SCRIPTS:
            fame_rows.append(r)
print(f"fame sym rows kept: {len(fame_rows)}")

# ── fame calligrapher: raw id -> 连续 id (id_map) ─────────────────────────
raw_ids = set()
callig_name2raw = {}
with open("assets/train_fame3_clean_v8.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        raw_ids.add(int(r["calligrapher_id"]))
        callig_name2raw[r["calligrapher"]] = int(r["calligrapher_id"])

# strict eval 泄漏对
strict_pairs = set()
with open("assets/eval_fame3_strict_clean_v9.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        strict_pairs.add((r["calligrapher"], r["script"], r["character"]))

# ── tongji 原始行 + v4 对称增强 ────────────────────────────────────────────
os.makedirs(TONGJI_SYM, exist_ok=True)
import glob
orig_rows = []
new_calligs = {}
skipped = {"grass": 0, "leak": 0}
iid = IMG_ID_BASE
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
            import shutil
            shutil.copy(p, dst)
        orig_rows.append({
            "image_path": f"{TONGJI_FLAT}/{iid}.png", "calligrapher": callig,
            "script": script, "character": ch, "calligrapher_id": cid,
            "script_id": {"楷": "0", "行": "3", "隶": "4"}[script],
            "character_id": csid, "glyph_id": gid, "aug": "",
            "_idx": iid - IMG_ID_BASE, "_iid": iid,
        })
        iid += 1
print(f"tongji originals: {len(orig_rows)}, skipped={skipped}")


def work(task):
    """v4 对称增强: ±同幅 dilate/erode, p 由 idx 哈希, 保护性减档。"""
    r = task
    try:
        g = np.asarray(Image.open(r["image_path"]).convert("L"))
    except Exception as e:
        return r, f"FAIL {r['image_path']}: {e}"
    ink = g < 128
    if ink.sum() < 20:
        return r, {"tp": None, "tn": None}
    p = 1 + ((r["_idx"] * 2654435761) % 2)
    out = {}
    for kind, base, op in (("tp", UID_T, binary_dilation), ("tn", UID_N, binary_erosion)):
        pp, ok = p, False
        while pp > 0:
            m = op(ink, ST, iterations=pp)
            a = int(m.sum())
            if kind == "tn" and (a < 0.15 * int(ink.sum()) or a < 20):
                pp -= 1
                continue
            if kind == "tp" and a > 3.0 * int(ink.sum()):
                pp -= 1
                continue
            ok = True
            break
        if not ok:
            out[kind] = None
            continue
        uid = base + r["_idx"]
        Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(
            os.path.join(TONGJI_SYM, f"{uid}.png"))
        out[kind] = uid
    return r, out


with Pool(32) as pool:
    results = list(pool.imap_unordered(work, orig_rows, chunksize=16))
fails = sum(1 for _, v in results if isinstance(v, str))
print(f"tongji v4 aug done: {len(results)} tasks, fails={fails}")

fields = ["image_path", "calligrapher", "script", "character", "calligrapher_id",
          "script_id", "character_id", "glyph_id", "aug"]
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in fame_rows:
        w.writerow({k: r[k] for k in fields})
    n_tj = 0
    for r, out in results:
        w.writerow({k: r[k] for k in fields})
        n_tj += 1
        for kind, uid in (out or {}).items():
            if uid is None:
                continue
            w.writerow({
                "image_path": f"{TONGJI_SYM}/{uid}.png", "calligrapher": r["calligrapher"],
                "script": r["script"], "character": r["character"],
                "calligrapher_id": r["calligrapher_id"], "script_id": r["script_id"],
                "character_id": r["character_id"], "glyph_id": r["glyph_id"],
                "aug": kind})
            n_tj += 1
print(f"written {OUT_CSV}: {len(fame_rows)} fame + {n_tj} tongji rows (orig+aug)")

# ── 扩展 callig id map ─────────────────────────────────────────────────────
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
print(f"written {OUT_MAP}: num_calligraphers={cmap['num_calligraphers']} "
      f"(new: {sorted(new_calligs)})")
