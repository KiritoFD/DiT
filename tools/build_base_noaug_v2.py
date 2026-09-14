# -*- coding: utf-8 -*-
"""
build_base_noaug_v2.py — base 无增强数据集 v2 (名归一 + 500 张/书家门槛).

  fame3 原始 (clean_v8, 楷0/行3/隶4) + tongji 原始 + UniCalli 裁切 (归一名)
  规则: 全库按书家合并统计; 总图数 <500 的书家整家剔除 (fame 原则), 报告名单;
        strict 泄漏排除; 新字 50000+ 段; 新书家 raw 9000+/9100+.
输出: assets/train_base_noaug.csv (覆盖 v1)
      assets/callig_id_map_base.json
      base_noaug_report.txt (保留/剔除书家名单)
"""
import collections as C
import csv
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_CSV = "assets/train_base_noaug.csv"
OUT_MAP = "assets/callig_id_map_base.json"
REPORT = "base_noaug_report.txt"
KEEP = {"楷": "0", "行": "3", "隶": "4"}
NAME_NORM = {"文征明": "文徵明", "朱耷\\八大山人": "朱耷"}
MIN_PER_CALLIG = 224
fields = ["image_path", "calligrapher", "script", "character", "calligrapher_id",
          "script_id", "character_id", "glyph_id", "aug"]

# ── fame 字表 + 名字映射 + 原始行 ──────────────────────────────────────────
char_map = {}
raw_ids = set()
callig_name2raw = {}
fame_rows = []
with open("assets/train_fame3_clean_v8.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        char_map.setdefault(r["character"], (r["character_id"], r["glyph_id"]))
        raw_ids.add(int(r["calligrapher_id"]))
        callig_name2raw[r["calligrapher"]] = int(r["calligrapher_id"])
        if r["script"] in KEEP:
            row = dict(r)
            row.setdefault("aug", "")
            fame_rows.append(row)

strict_pairs = set()
with open("assets/eval_fame3_strict_clean_v9.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        strict_pairs.add((r["calligrapher"], r["script"], r["character"]))

# ── tongji 原始 (复用 v1 已平铺的 calli_tongji_imgs: 按内容重新映射) ───────
tongji_src = {}   # (callig, script, char) -> flat path (v1 平铺文件仍在)
for p in glob.glob("data/imgs/calli_tongji_imgs/*.png"):
    iid = int(os.path.basename(p)[:-4])
    if 950000 <= iid < 953000:
        tongji_src.setdefault(iid, p)
# v1 flat 是按 950000+ 顺序复制的: 重建 (callig,script,char)->flat 需要目录结构
flat_by_key = {}
import hashlib
rows_v1 = list(csv.DictReader(open("assets/train_fame_tj_kxl.csv", encoding="utf-8")))
v1_map = {}
for r in rows_v1:
    if "calli_tongji" in r["image_path"] and r["aug"] == "":
        v1_map[(r["calligrapher"], r["script"], r["character"])] = r["image_path"]

rows_all = []
skipped = C.Counter()
for r in fame_rows:
    rows_all.append(dict(r, _src="fame"))

for (callig, script, ch), img_path in v1_map.items():
    if (callig, script, ch) in strict_pairs:
        skipped["tongji_leak"] += 1
        continue
    if ch not in char_map:
        char_map[ch] = (str(50000 + len(char_map)), str(50000 + len(char_map)))
    rows_all.append({"image_path": img_path, "calligrapher": callig,
                     "script": script, "character": ch, "script_id": KEEP[script],
                     "aug": "", "_src": "tongji"})

with open("assets/unicalli_rows.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        callig = NAME_NORM.get(r["calligrapher"], r["calligrapher"])
        ch = r["character"]
        if (callig, r["script"], ch) in strict_pairs:
            skipped["uc_leak"] += 1
            continue
        if ch not in char_map:
            char_map[ch] = (str(50000 + len(char_map)), str(50000 + len(char_map)))
        rows_all.append({"image_path": r["image_path"], "calligrapher": callig,
                         "script": r["script"], "character": ch,
                         "script_id": r["script_id"], "aug": "", "_src": "unicalli"})

# ── <500 剔除 ──────────────────────────────────────────────────────────────
cnt = C.Counter(r["calligrapher"] for r in rows_all)
dropped = {c: v for c, v in cnt.items() if v < MIN_PER_CALLIG}
kept_rows = [r for r in rows_all if cnt[r["calligrapher"]] >= MIN_PER_CALLIG]

new_calligs = {}
for r in kept_rows:
    c = r["calligrapher"]
    if c not in callig_name2raw:
        callig_name2raw[c] = 9000 + len(new_calligs)
        new_calligs[c] = 9000 + len(new_calligs)

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in kept_rows:
        out = {k: r[k] for k in fields if k in r}
        if r["_src"] != "fame":
            out["calligrapher_id"] = str(callig_name2raw[r["calligrapher"]])
            csid, gid = char_map[r["character"]]
            out["character_id"], out["glyph_id"] = csid, gid
        else:
            out["calligrapher_id"] = str(callig_name2raw[r["calligrapher"]])
        w.writerow({k: out.get(k, "") for k in fields})

kept_cnt = C.Counter(r["calligrapher"] for r in kept_rows)
rep = open(REPORT, "w", encoding="utf-8")
rep.write(f"kept rows: {len(kept_rows)} (fame {len(fame_rows)}, "
          f"tongji {sum(1 for r in kept_rows if r['_src']=='tongji')}, "
          f"unicalli {sum(1 for r in kept_rows if r['_src']=='unicalli')})\n")
rep.write(f"skipped: {dict(skipped)}\n\n")
rep.write(f"dropped calligraphers (<{MIN_PER_CALLIG} 张): {len(dropped)}\n")
for c, v in sorted(dropped.items(), key=lambda x: -x[1]):
    rep.write(f"  {c}: {v}\n")
rep.write(f"\nkept calligraphers ({len(kept_cnt)}):\n")
for c, v in kept_cnt.most_common():
    srcs = set()
    for r in kept_rows:
        if r["calligrapher"] == c:
            srcs.add(r["_src"])
    rep.write(f"  {c}: {v} ({'+'.join(sorted(srcs))})\n")
rep.close()

with open("assets/callig_id_map.json", encoding="utf-8") as f:
    cmap = json.load(f)
id_map = {str(k): int(v) for k, v in cmap.get("id_map", {}).items()}
n_cont = int(cmap.get("num_calligraphers", 41))
for callig, new_raw in sorted(new_calligs.items()):
    id_map[str(new_raw)] = n_cont + list(new_calligs).index(callig)
cmap["id_map"] = id_map
cmap["num_calligraphers"] = n_cont + len(new_calligs)
with open(OUT_MAP, "w", encoding="utf-8") as f:
    json.dump(cmap, f, ensure_ascii=False, indent=2)
print(f"kept {len(kept_rows)} rows, dropped {len(dropped)} calligraphers (<{MIN_PER_CALLIG}), "
      f"num_calligraphers={cmap['num_calligraphers']} (new {len(new_calligs)})")
