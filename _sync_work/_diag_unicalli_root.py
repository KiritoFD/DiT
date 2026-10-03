# -*- coding: utf-8 -*-
"""_diag_unicalli_root.py — 定位 unicalli 坏图是哪一步产生的.

复现 crop_unicalli.py 的编号顺序 (img_id = 960000 + k, k 为全局 crop 序号),
对坏图 id 反查 (原图路径, bbox), 检查:
  a) bbox 是否越界           -> PIL crop 越界填黑 = 黑块
  b) 原图是否黑底            -> 整块黑
  c) crop 区域自身灰度       -> 区分"内容糊" vs "填充黑"
"""
import ast
import csv
import os
import sys

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

# 读入坏图 id 列表
bad_ids = []
if os.path.exists("/tmp/bad_imgs.txt"):
    for line in open("/tmp/bad_imgs.txt", encoding="utf-8"):
        p = line.split("\t")[0].strip()
        if p.endswith(".png"):
            bad_ids.append(int(os.path.basename(p)[:-4]))
bad_set = set(bad_ids)
print(f"[bad] {len(bad_set)} 张坏图 id, 范围 {min(bad_set)}~{max(bad_set)}", flush=True)

# 复现 crop_unicalli 的编号顺序
rows = list(csv.DictReader(open("data/unicalli/data.csv", encoding="utf-8")))
k = 0
targets = {}
n_bbox_oob = 0
n_total = 0
stats_black_crop = 0
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
        if iid in bad_set:
            targets[iid] = (r["img_path"], tuple(b["p"]), ch, author, chiro)
        k += 1

print(f"[map] 总 crop 序号={k}, 命中坏图={len(targets)}", flush=True)

# 逐个检查原图
checked = 0
for iid in sorted(targets)[:40]:
    img_path, (x1, y1, x2, y2), ch, author, chiro = targets[iid]
    full = os.path.join(IMG_ROOT, img_path)
    if not os.path.exists(full):
        print(f"  {iid} {ch} {author}: 原图不存在 {full}")
        continue
    im = Image.open(full).convert("L")
    W, H = im.size
    a = np.asarray(im, dtype=np.float32)
    oob = (x1 < 0 or y1 < 0 or x2 > W or y2 > H)
    # crop 实际区域(裁剪到合法范围)
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x2), min(H, y2)
    sub = a[cy1:cy2, cx1:cx2] if (cx2 > cx1 and cy2 > cy1) else np.zeros((1, 1), np.float32)
    # PIL crop 越界会填 0 -> 模拟
    print(f"  id={iid} '{ch}' {author}/{chiro}")
    print(f"      原图 {W}x{H} mean={a.mean():.1f} | bbox=({x1},{y1},{x2},{y2}) "
          f"OOB={oob} | crop区 mean={sub.mean():.1f} fg={(sub<127).mean():.3f}")
    if oob:
        n_bbox_oob += 1
    if sub.mean() < 120 and (sub < 127).mean() > 0.5:
        stats_black_crop += 1
    checked += 1

print(f"\n[子集统计] 检查 {checked}: bbox越界={n_bbox_oob}, crop区本身黑={stats_black_crop}", flush=True)

# 全量统计: 有多少 bbox 越界 / 原图黑底
print("\n[全量] 扫描全部 crop 的 bbox 越界与 crop 区黑度 (可能较慢) ...", flush=True)
k = 0
oob_cnt = black_cnt = tot = 0
oob_ex = []
black_ex = []
from collections import defaultdict
by_img_cache = {}
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
        x1, y1, x2, y2 = b["p"]
        full = os.path.join(IMG_ROOT, r["img_path"])
        if full not in by_img_cache:
            try:
                by_img_cache[full] = Image.open(full).convert("L")
                by_img_cache[full].load()
            except Exception:
                by_img_cache[full] = None
        im = by_img_cache[full]
        tot += 1
        if im is None:
            continue
        W, H = im.size
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            oob_cnt += 1
            if len(oob_ex) < 5:
                oob_ex.append((IMG_ID_BASE + k, ch, r["img_path"], (x1, y1, x2, y2), (W, H)))
        k += 1
        if tot % 20000 == 0:
            print(f"    {tot} ... oob={oob_cnt}", flush=True)

print(f"\n  总 crop={tot}  bbox越界={oob_cnt} ({100*oob_cnt/max(tot,1):.2f}%)")
for e in oob_ex:
    print(f"    例: id={e[0]} '{e[1]}' {e[2]} bbox={e[3]} 原图={e[4]}")
