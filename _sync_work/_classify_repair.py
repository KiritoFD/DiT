# -*- coding: utf-8 -*-
"""_classify_repair.py — 对坏图分类(黑底原图 / bbox越界 / 真糊), 并导出"修复后"预览.

修复思路:
  1) bbox 越界 -> clamp 到合法范围 (PIL crop 越界会填黑)
  2) 底色检测 -> 若 crop 区中位数 < 128 判为黑底(拓片), 反相 255-x
  3) 仍为"无结构"的 -> 无救, 建议删除
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
OUT = "/root/Workspace/xy/DiT/_otout_blob"
os.makedirs(OUT, exist_ok=True)

bad_set = set()
for line in open("/tmp/bad_imgs.txt", encoding="utf-8"):
    p = line.split("\t")[0].strip()
    if p.endswith(".png"):
        bad_set.add(int(os.path.basename(p)[:-4]))
print(f"[bad] {len(bad_set)}", flush=True)

# id -> (img_path, bbox, char, author, chiro)
rows = list(csv.DictReader(open("data/unicalli/data.csv", encoding="utf-8")))
targets = {}
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
        if iid in bad_set:
            targets[iid] = (r["img_path"], tuple(int(v) for v in b["p"]), ch, author, chiro)
        k += 1
print(f"[map] 命中 {len(targets)}/{len(bad_set)}", flush=True)

# 缓存原图
cache = {}


def get_im(p):
    if p not in cache:
        try:
            im = Image.open(os.path.join(IMG_ROOT, p)).convert("L")
            im.load()
            cache[p] = im
        except Exception:
            cache[p] = None
    return cache[p]


cls = {"ok_after_fix": 0, "oob_only": 0, "dark_need_invert": 0, "both": 0,
       "hopeless": 0, "missing": 0}
samples = []
for iid in sorted(targets):
    img_path, (x1, y1, x2, y2), ch, author, chiro = targets[iid]
    im = get_im(img_path)
    if im is None:
        cls["missing"] += 1
        continue
    W, H = im.size
    oob = (x1 < 0 or y1 < 0 or x2 > W or y2 > H)
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x2), min(H, y2)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        cls["hopeless"] += 1
        continue
    sub = np.asarray(im.crop((cx1, cy1, cx2, cy2)), dtype=np.float32)
    med = float(np.median(sub))
    dark = med < 128
    # 修复: clamp + 反相
    fix = 255.0 - sub if dark else sub
    fg_fix = float((fix < 127).mean())
    if oob and dark:
        cls["both"] += 1
    elif oob:
        cls["oob_only"] += 1
    elif dark:
        cls["dark_need_invert"] += 1
    if 0.02 < fg_fix < 0.6:
        cls["ok_after_fix"] += 1
    elif fg_fix >= 0.6:
        cls["hopeless"] += 1
    if len(samples) < 12:
        samples.append((iid, ch, author, img_path, (x1, y1, x2, y2), (W, H),
                        oob, med, fg_fix, fix))

print("\n=== 分类 ===")
for kk, vv in cls.items():
    print(f"  {kk:20s} {vv}")

print("\n=== 样例 (id, 字, 原图尺寸, bbox越界, crop中位数, 修复后fg) ===")
for (iid, ch, author, ip, bb, WH, oob, med, fgf, _) in samples[:12]:
    print(f"  {iid} '{ch}' {author:6s} {WH[0]}x{WH[1]} oob={oob} med={med:5.1f} fix_fg={fgf:.3f}")

# 导出预览: 原crop(带越界黑) vs 修复后 256
for n, (iid, ch, author, ip, bb, WH, oob, med, fgf, fix) in enumerate(samples[:8]):
    im = get_im(ip)
    W, H = im.size
    x1, y1, x2, y2 = bb
    # 原(含越界黑填充)
    raw = np.asarray(im.crop((x1, y1, x2, y2)), dtype=np.float32)
    Image.fromarray(raw.astype(np.uint8)).resize((256, 256), Image.LANCZOS).save(
        f"{OUT}/{n:02d}_raw_id{iid}_{ch}.png")
    # 修复后
    Image.fromarray(fix.astype(np.uint8)).resize((256, 256), Image.LANCZOS).save(
        f"{OUT}/{n:02d}_fix_id{iid}_{ch}.png")
print(f"\n-> 预览: {OUT}")
