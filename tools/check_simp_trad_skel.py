"""检测简繁错配 —— 用**骨架 IoU**，不依赖 OCR。

## 原理
GT 是书法家写的字，风格差异大，但**拓扑（骨架）**是稳定的。
对每个样本:
    GT 骨架  vs  std[当前 character] 骨架   -> IoU_now
    GT 骨架  vs  std[繁体/简体变体] 骨架     -> IoU_alt
如果 IoU_alt 明显 > IoU_now -> 该样本简繁标错了。

## 为什么比 OCR 可靠
- 骨架只保留笔画拓扑，去掉粗细/风格 -> 对行草也稳
- 不需要 OCR 的 3755 字表限制
- 可解释（IoU 数值）

输出: assets/simp_trad_mismatch.csv
    image_path, script, calligrapher, char_csv, char_alt, iou_now, iou_alt, verdict
"""
import argparse
import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="0=全部")
ap.add_argument("--script", default="all")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", default="assets/simp_trad_mismatch.csv")
a = ap.parse_args()

from opencc import OpenCC

s2t = OpenCC("s2t")
t2s = OpenCC("t2s")

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
if a.script != "all":
    rows = [r for r in rows if r["script"] == a.script]
if a.limit:
    rows = rows[:a.limit]
print(f"  样本: {len(rows)}")

# char -> std_path（每个字取第一条）
std_of = {}
for r in rows:
    std_of.setdefault(r["character"], r["std_path"])
print(f"  字表: {len(std_of)}")


def to_skel(path, size=128):
    """读图 -> 二值 -> 骨架 -> **裁剪到笔画包围盒并缩放到 size** -> bool 数组。

    ⚠ 必须做包围盒归一化（实测）: GT 书法字和 std 印刷体字的**位置/尺度不同**，
       直接 resize 到同一尺寸算 IoU 只有 0.01~0.05（几乎不重合，无意义）。
       先裁到笔画包围盒再缩放到同尺寸，IoU 才有可比性。
    """
    p = path if os.path.isabs(path) else os.path.join("/root/Workspace/xy/DiT", path)
    im = Image.open(p).convert("L")
    arr0 = np.array(im, dtype=np.float32)
    thr0 = arr0.mean() - arr0.std() * 0.5
    bw0 = arr0 < thr0
    if bw0.sum() < 20:          # 反色情况
        bw0 = arr0 > thr0
    ys, xs = np.where(bw0)
    if len(ys) < 10:
        return np.zeros((size, size), dtype=bool)
    crop = im.crop((int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1))
    # 保持长宽比缩放到 size，再居中放到 size x size
    w, h = crop.size
    sc = size / max(w, h)
    crop = crop.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(crop, ((size - crop.size[0]) // 2, (size - crop.size[1]) // 2))
    arr = np.array(canvas, dtype=np.float32)
    thr = arr.mean() - arr.std() * 0.5
    bw = arr < thr
    if bw.sum() < 20:
        bw = arr > thr
    return skeletonize(bw)


def iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union else 0.0


# 预计算 std 骨架（含简繁变体）
std_cache = {}
need = set()
for c in std_of:
    need.add(c)
    t = s2t.convert(c)
    if t != c:
        need.add(t)
    s = t2s.convert(c)
    if s != c:
        need.add(s)
print(f"  需要 std 骨架: {len(need)}")


def _mk(c):
    p = std_of.get(c)
    if not p:
        return c, None
    try:
        return c, to_skel(p)
    except Exception:
        return c, None


with ThreadPoolExecutor(a.workers) as ex:
    for c, sk in ex.map(_mk, sorted(need)):
        std_cache[c] = sk
print(f"  已建 std 骨架: {sum(1 for v in std_cache.values() if v is not None)}")


def check(r):
    ch = r["character"]
    alt = s2t.convert(ch)
    if alt == ch:
        alt = t2s.convert(ch)
    if alt == ch:
        return None
    sk_gt = None
    try:
        sk_gt = to_skel(r["image_path"])
    except Exception:
        return None
    sk_now = std_cache.get(ch)
    sk_alt = std_cache.get(alt)
    if sk_now is None or sk_alt is None:
        return None
    i_now = iou(sk_gt, sk_now)
    i_alt = iou(sk_gt, sk_alt)
    return (r["image_path"], r["script"], r["calligrapher"], ch, alt,
            round(i_now, 4), round(i_alt, 4))


from time import time

t0 = time()
res = []
with ThreadPoolExecutor(a.workers) as ex:
    for i, v in enumerate(ex.map(check, rows)):
        if v is not None:
            res.append(v)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(rows)}  {time()-t0:.0f}s", flush=True)
print(f"  完成 {len(res)} 条有简繁变体的样本, {time()-t0:.0f}s")

# 判定: alt 明显更好
TH = 0.03
mismatch = [r for r in res if r[6] - r[5] > TH]
print(f"\n  === 结果 ===")
print(f"    有简繁变体的样本: {len(res)}")
print(f"    alt 明显更好(IoU 差 >{TH}): {len(mismatch)} = "
      f"{len(mismatch)/max(len(res),1)*100:.1f}%")

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "script", "calligrapher", "char_csv",
                "char_alt", "iou_now", "iou_alt", "delta"])
    for r in sorted(res, key=lambda x: -(x[6] - x[5])):
        w.writerow(list(r) + [round(r[6] - r[5], 4)])
print(f"  written {a.out}")

print(f"\n  === 最可疑的 20 条 ===")
for r in sorted(res, key=lambda x: -(x[6] - x[5]))[:20]:
    print(f"    csv={r[3]}  alt={r[4]}  IoU {r[5]:.3f}->{r[6]:.3f}  "
          f"[{r[1]}/{r[2]}]")
