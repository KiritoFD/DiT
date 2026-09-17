# -*- coding: utf-8 -*-
"""rollback_v4.py — 回滚 purge_clean_v1_r2 的剔除, 并重建完整 csv."""
import csv
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUAR = "data/_quarantine_v4"
IMGS = "data/clean_v1/imgs"
STD = "data/clean_v1/std"

n = 0
for f in glob.glob(os.path.join(QUAR, "*.png")):
    dst = os.path.join(IMGS, os.path.basename(f))
    if not os.path.exists(dst):
        shutil.move(f, dst)
        n += 1
print(f"[rollback] restored {n} -> {IMGS}")
print(f"  imgs now: {len(glob.glob(os.path.join(IMGS,'*.png')))}")
print(f"  std  now: {len(glob.glob(os.path.join(STD,'*.png')))}")

# 用 imgs+std 都存在 的 id 重建 csv
std_ids = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(STD, "*.png"))}
rows_out = []
for p in sorted(glob.glob(os.path.join(IMGS, "*.png"))):
    iid = os.path.basename(p)[:-4]
    if iid not in std_ids:
        continue
    rows_out.append({"image_path": f"{IMGS}/{iid}.png", "std_path": f"{STD}/{iid}.png"})
# 与原始 csv 合并元数据 (character/script/calligrapher)
meta = {}
for src in ("assets/train_clean_v1.csv", "assets/train_base_clean.csv"):
    if not os.path.exists(src):
        continue
    for r in csv.DictReader(open(src, encoding="utf-8")):
        iid = os.path.basename(r["image_path"])[:-4]
        meta.setdefault(iid, r)
fields = ["image_path", "calligrapher", "script", "character", "calligrapher_id",
          "script_id", "character_id", "glyph_id", "aug", "std_path"]
out = "assets/train_clean_v1_final.csv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows_out:
        iid = os.path.basename(r["image_path"])[:-4]
        m = dict(meta.get(iid, {}))
        m["image_path"] = r["image_path"]
        m["std_path"] = r["std_path"]
        m.setdefault("aug", "")
        w.writerow(m)
print(f"[rollback] {out}: {len(rows_out)} 行")
