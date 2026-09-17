# -*- coding: utf-8 -*-
"""check_tongji_extra.py — 查 noaug 未收录的 900 张 tongji 是什么."""
import csv
import glob
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "/root/Workspace/xy/DiT/_otout_tj_extra"
os.makedirs(OUT, exist_ok=True)

noaug = set()
for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
    if "tongji" in r["image_path"]:
        noaug.add(os.path.basename(r["image_path"])[:-4])

allf = sorted(glob.glob("data/imgs/calli_tongji_imgs/*.png"))
extra = [p for p in allf if os.path.basename(p)[:-4] not in noaug]
print(f"calli_tongji_imgs 共 {len(allf)}, noaug 收录 {len(noaug)}, 未收录 {len(extra)}")

# 与 tongji_only.csv 对照元数据
meta = {}
for r in csv.DictReader(open("assets/train_tongji_only.csv", encoding="utf-8")):
    meta[os.path.basename(r["image_path"])[:-4]] = r
hit = [p for p in extra if os.path.basename(p)[:-4] in meta]
print(f"  其中在 tongji_only.csv 有元数据的: {len(hit)}")
if hit:
    rr = [meta[os.path.basename(p)[:-4]] for p in hit]
    print(f"  书家: {dict(Counter(r.get('calligrapher','') for r in rr).most_common(12))}")
    print(f"  书体: {dict(Counter(r.get('script','') for r in rr))}")
    print(f"  字符数: {len({r.get('character','') for r in rr})}")


def stat(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (p, -1.0)
    return (p, float((a < 128).mean()))


with mp.Pool(32) as pool:
    res = pool.map(stat, extra, chunksize=64)
ink = np.array([r[1] for r in res])
print(f"\n  这 {len(extra)} 张 ink: p1={np.percentile(ink,1):.4f} "
      f"p50={np.percentile(ink,50):.4f} p99={np.percentile(ink,99):.4f}")
print(f"    blank(<0.01)={int((ink<0.01).sum())}  过重(>0.6)={int((ink>0.6).sum())}")

CELL, COLS = 110, 8
sel = res[::max(1, len(res) // 64)][:64]
cv = Image.new("RGB", (COLS * CELL, ((len(sel) + COLS - 1) // COLS) * CELL), (200, 30, 30))
for i, (p, _) in enumerate(sel):
    try:
        im = Image.open(p).convert("L").resize((CELL, CELL))
        cv.paste(im.convert("RGB"), ((i % COLS) * CELL, (i // COLS) * CELL))
    except Exception:
        pass
cv.save(f"{OUT}/extra.png")
print(f"  -> {OUT}/extra.png ({len(sel)} 张)")

# 同时导出 noaug 里tongji 的抽样, 对照
sel2 = [r for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8"))
        if "tongji" in r["image_path"]][::max(1, len(noaug) // 64)][:64]
cv2 = Image.new("RGB", (COLS * CELL, ((len(sel2) + COLS - 1) // COLS) * CELL), (30, 30, 200))
for i, r in enumerate(sel2):
    try:
        im = Image.open(r["image_path"]).convert("L").resize((CELL, CELL))
        cv2.paste(im.convert("RGB"), ((i % COLS) * CELL, (i // COLS) * CELL))
    except Exception:
        pass
cv2.save(f"{OUT}/noaug.png")
print(f"  -> {OUT}/noaug.png ({len(sel2)} 张)")
