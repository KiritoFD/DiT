# -*- coding: utf-8 -*-
"""_scan_polarity_v2.py — 准确统计"黑底图": 死黑(必删) vs 反色(可救).

要点: crop_unicalli 把窄长条 crop 贴到 **白色 canvas** 上 (pad 填 255),
      旧判据被白 pad 掩盖 -> 漏检。这里先**去掉纯白 pad**取内容 bbox,
      再看**中心区**灰度众数(背景色) 判底色。

判据:
  c  = 内容区 (剔除 255 pad 的 bounding box)
  cc = c 的中心 50% 区域
  bg = mode(cc)
  bg >= 128                -> ok        (白底, 无需处理)
  bg <  128 (反色图):
       fg_inv = (cc > 128).mean()      # 反相后前景(笔画)占比
       fg_inv < 0.02        -> dead_black  (几乎无字形, 只能删)
       fg_inv > 0.85        -> odd         (反相后几乎全墨, 需人工看)
       其它                  -> invertible  (反相可救)
输出: /tmp/polarity_v2_<tag>.csv + 样例图
"""
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

CSV = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "base"
SAMPLE_DIR = f"/root/Workspace/xy/DiT/_otout_pol_{TAG}"


def probe(r):
    p = r.get("image_path", "")
    meta = (r.get("character", ""), r.get("calligrapher", ""), r.get("script", ""),
            r.get("aug", "") or "-")
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception as e:
        return (p, *meta, -1, 0.0, 0.0, "read_error")
    # 1) 去纯白 pad: 找非白内容的 bbox
    m = a < 250
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    if not rows.any():
        return (p, *meta, 255, 0.0, 0.0, "dead_black")     # 整幅全白 -> 无内容
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    c = a[r0:r1 + 1, c0:c1 + 1]
    # 2) 中心 50% 区域
    h, w = c.shape
    cc = c[h // 4:max(h // 4 + 1, 3 * h // 4), w // 4:max(w // 4 + 1, 3 * w // 4)]
    if cc.size < 64:
        cc = c
    hh = np.bincount(cc.ravel(), minlength=256)
    bg = int(hh.argmax())
    if bg >= 128:
        return (p, *meta, bg, float((cc < 128).mean()), 0.0, "ok")
    fg_inv = float((cc > 128).mean())          # 反相后前景占比
    if fg_inv < 0.02:
        v = "dead_black"
    elif fg_inv > 0.85:
        v = "odd"
    else:
        v = "invertible"
    return (p, *meta, bg, fg_inv, float((cc < 128).mean()), v)


def main():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"[scan] {CSV}: {len(rows)} rows", flush=True)
    t0 = time.time()
    res = []
    with mp.Pool(40) as pool:
        for n, rr in enumerate(pool.imap_unordered(probe, rows, chunksize=256), 1):
            res.append(rr)
            if n % 20000 == 0:
                print(f"  {n}/{len(rows)} ({n/(time.time()-t0):.0f}/s)", flush=True)

    cnt = Counter(r[-1] for r in res)
    tot = len(res)
    print(f"\n=== {TAG} 结果 ({time.time()-t0:.0f}s, {tot} 行) ===")
    for k in ("ok", "invertible", "dead_black", "odd", "read_error"):
        c = cnt.get(k, 0)
        print(f"  {k:12s} {c:7d}  ({100*c/tot:6.2f}%)")
    bad = cnt.get("invertible", 0) + cnt.get("dead_black", 0) + cnt.get("odd", 0)
    print(f"  {'黑底合计':12s} {bad:7d}  ({100*bad/tot:6.2f}%)")

    # 数据源 / 唯一图统计
    def src_of(p):
        parts = p.split("/")
        return parts[2] if len(parts) > 2 else "?"
    by = {}
    uniq = {"invertible": set(), "dead_black": set(), "odd": set()}
    for r in res:
        s = src_of(r[0])
        d = by.setdefault(s, Counter())
        d[r[-1]] += 1
        if r[-1] in uniq:
            uniq[r[-1]].add(r[0])
    print("\n  按数据源 (行数):")
    for s, d in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        print(f"    {s:22s} 总={sum(d.values()):7d} "
              f"可救={d.get('invertible',0):6d} 死黑={d.get('dead_black',0):5d} "
              f"odd={d.get('odd',0):4d}")
    print("\n  唯一图数: "
          + "  ".join(f"{k}={len(v)}" for k, v in uniq.items()))

    # 死黑样例
    dead = [r for r in res if r[-1] == "dead_black"]
    print(f"\n  死黑样例 ({len(dead)}):")
    for r in dead[:12]:
        print(f"    {r[0]} bg={r[4]:3d} 反相后fg={r[5]:.4f} char={r[1]} {r[2]}")
    odd = [r for r in res if r[-1] == "odd"]
    print(f"\n  odd 样例 ({len(odd)}):")
    for r in odd[:8]:
        print(f"    {r[0]} bg={r[4]:3d} 反相后fg={r[5]:.4f} char={r[1]} {r[2]}")

    # 导出样例图
    for name, kind in (("inv", "invertible"), ("dead", "dead_black"), ("odd", "odd")):
        sel = [r for r in res if r[-1] == kind][:6]
        for i, r in enumerate(sel):
            try:
                Image.open(r[0]).convert("L").resize((256, 256)).save(
                    f"{SAMPLE_DIR}/{name}{i}_bg{r[4]}_fg{r[5]:.3f}.png")
            except Exception:
                pass
    print(f"  -> 样例: {SAMPLE_DIR}")

    with open(f"/tmp/polarity_v2_{TAG}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "character", "calligrapher", "script", "aug",
                    "bg_mode", "fg_after_invert", "ink_ratio", "verdict"])
        for r in res:
            if r[-1] != "ok":
                w.writerow(r)
    print(f"  -> /tmp/polarity_v2_{TAG}.csv")


if __name__ == "__main__":
    main()
