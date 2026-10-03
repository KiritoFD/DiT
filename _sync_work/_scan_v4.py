# -*- coding: utf-8 -*-
"""_scan_v4.py — 精确判定: 死黑(必删) vs 反色(可救).

关键: 下游 PNG 含**白 pad**, 会大幅抬高 bright 使判据失真。
     这里直接从 UniCalli **原图 + bbox** 重裁 (clamp), 得到无 pad 的真实内容,
     再判底色与字形含量。

输出: 可救/死黑 计数 + 名称清单 + montage
"""
import ast
import csv
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME_MAP = {"赵佶\\宋徽宗": "宋徽宗", "文征明": "文徵明"}
EXCLUDE_AUTHORS = {"佚名", "摩崖刻石", "墓志", "造像记", "金文刻石", "碑刻", "墨迹"}
KEEP = {"楷": "0", "行": "3", "隶": "4"}
IMG_ID_BASE = 960000
IMG_ROOT = "data/unicalli/images"
OUT = "/root/Workspace/xy/DiT/_otout_v4"
os.makedirs(OUT, exist_ok=True)

INK_HI = 0.50
DEAD = 0.03          # 反相后前景 <3% -> 无字形

# 候选 = v3 判为黑底的图
cand = set()
for r in csv.DictReader(open("/tmp/v3_base.csv", encoding="utf-8")):
    cand.add(r["image_path"])
print(f"[cand] v3 黑底候选 {len(cand)}", flush=True)

# 重建 unicalli id -> (原图, bbox, char, author, chiro)
rows = list(csv.DictReader(open("data/unicalli/data.csv", encoding="utf-8")))
id2info = {}
k = 0
for r in rows:
    boxes = ast.literal_eval(r["location"])
    author = NAME_MAP.get(r["author"], r["author"].strip())
    chiro = r["chirography"].strip()
    if author in EXCLUDE_AUTHORS or chiro not in KEEP:
        continue
    for b in boxes:
        ch = b["c"]
        if not (len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"):
            continue
        iid = IMG_ID_BASE + k
        if f"data/imgs/unicalli_chars/{iid}.png" in cand:
            id2info[iid] = (r["img_path"], tuple(int(v) for v in b["p"]), ch, author, chiro)
        k += 1
print(f"[map] unicalli 命中 {len(id2info)} / {len(cand)}", flush=True)

_cache = {}


def get_im(p):
    if p not in _cache:
        try:
            im = Image.open(os.path.join(IMG_ROOT, p)).convert("L")
            im.load()
            _cache[p] = im
        except Exception:
            _cache[p] = None
    return _cache[p]


def probe(item):
    iid, (img_path, bb, ch, author, chiro) = item
    im = get_im(img_path)
    if im is None:
        return (iid, img_path, bb, ch, author, chiro, 0.0, 0.0, 0.0, "missing")
    W, H = im.size
    x1, y1, x2, y2 = bb
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return (iid, img_path, bb, ch, author, chiro, 1.0, 0.0, 0.0, "tiny")
    sub = np.asarray(im.crop((x1, y1, x2, y2)), dtype=np.uint8)
    ink = float((sub < 128).mean())
    bright = float((sub > 128).mean())
    med = float(np.median(sub))
    if ink <= INK_HI:
        v = "ok"                      # 其实是白底浓墨 -> 误报, 不动
    elif bright < DEAD:
        v = "dead_black"
    elif bright <= 0.50:
        v = "invertible"
    else:
        v = "odd"
    return (iid, img_path, bb, ch, author, chiro, ink, bright, med, v)


def main():
    items = sorted(id2info.items())
    t0 = time.time()
    res = []
    with mp.Pool(32) as pool:
        for n, rr in enumerate(pool.imap_unordered(probe, items, chunksize=16), 1):
            res.append(rr)
            if n % 1000 == 0:
                print(f"  {n}/{len(items)} ({n/(time.time()-t0):.0f}/s)", flush=True)

    cnt = Counter(r[-1] for r in res)
    print(f"\n=== unicalli 候选 {len(res)} 张判定 ({time.time()-t0:.0f}s) ===")
    for kk in ("ok", "invertible", "dead_black", "odd", "tiny", "missing"):
        print(f"  {kk:12s} {cnt.get(kk,0):6d}")

    print(f"\n  死黑样例 (反相后前景 <{DEAD}):")
    for r in [x for x in res if x[-1] == "dead_black"][:12]:
        print(f"    {r[0]} ink={r[6]:.3f} 反相后={r[7]:.4f} med={r[8]:.0f} "
              f"char={r[3]} {r[4]}")
    print(f"\n  可救样例 (反相后前景 {DEAD}~0.5):")
    for r in [x for x in res if x[-1] == "invertible"][:12]:
        print(f"    {r[0]} ink={r[6]:.3f} 反相后={r[7]:.3f} med={r[8]:.0f} "
              f"char={r[3]} {r[4]}")

    # 可救 vs 死黑 的 前景 分布
    inv = np.array([x[7] for x in res if x[-1] == "invertible"])
    dea = np.array([x[7] for x in res if x[-1] == "dead_black"])
    if inv.size:
        print(f"\n  可救 bright 分布: p10={np.percentile(inv,10):.3f} "
              f"p50={np.percentile(inv,50):.3f} p90={np.percentile(inv,90):.3f}")
    if dea.size:
        print(f"  死黑 bright 分布: p50={np.percentile(dea,50):.4f} "
              f"p90={np.percentile(dea,90):.4f} max={dea.max():.4f}")

    # montage: 原crop vs 反相后
    def montage(mode, out, sel):
        CELL, COLS = 96, 8
        n = min(len(sel), 64)
        nrow = max((n + COLS - 1) // COLS, 1)
        cv = Image.new("RGB", (COLS * CELL, nrow * CELL), (128, 0, 0))
        for i, r in enumerate(sel[:n]):
            im = get_im(r[1])
            if im is None:
                continue
            W, H = im.size
            x1, y1, x2, y2 = r[2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            a = np.asarray(im.crop((x1, y1, x2, y2)).resize((CELL, CELL)))
            if mode == "inv":
                a = 255 - a
            cv.paste(Image.fromarray(a).convert("RGB"),
                     ((i % COLS) * CELL, (i // COLS) * CELL))
        cv.save(out)

    montage("raw", f"{OUT}/raw.png", [x for x in res if x[-1] == "invertible"])
    montage("inv", f"{OUT}/inv.png", [x for x in res if x[-1] == "invertible"])
    montage("raw", f"{OUT}/dead_raw.png", [x for x in res if x[-1] == "dead_black"])
    montage("inv", f"{OUT}/dead_inv.png", [x for x in res if x[-1] == "dead_black"])
    print(f"  -> montage {OUT}")

    with open("/tmp/v4_verdict.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "img_path", "x1", "y1", "x2", "y2", "char",
                    "author", "script", "ink", "bright_after_invert", "median", "verdict"])
        for r in res:
            w.writerow([r[0], r[1], *r[2], r[3], r[4], r[5], f"{r[6]:.4f}",
                        f"{r[7]:.4f}", f"{r[8]:.0f}", r[9]])
    print("  -> /tmp/v4_verdict.csv")


if __name__ == "__main__":
    main()
