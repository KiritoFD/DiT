# -*- coding: utf-8 -*-
"""stat_final_and_60px.py — 统计 fame3+tongji(2092) 的规模, 并测 60px 判据影响.

输出:
  * 书法家数 / 唯一字符数 / 唯一(script,char) / 书体分布 / 每字样本数分布
  * w_max > 60px 判据会剔掉多少 (整体 + 按源)
"""
import csv
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KEEP = ("final_imgs_fame_v8", "calli_tongji_imgs")
TH = 60.0


def probe(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return -1.0
    m = a < 128
    if not m.any():
        return 0.0
    return float(distance_transform_edt(m).max()) * 2.0


def main():
    rows = [r for r in csv.DictReader(
        open("assets/train_base_noaug.csv", encoding="utf-8"))
        if any(s in r["image_path"] for s in KEEP)]
    print(f"=== 数据集规模 ===")
    print(f"  样本数          : {len(rows)}")
    print(f"  书法家数        : {len({r.get('calligrapher','') for r in rows})}")
    print(f"  唯一 character  : {len({r.get('character','') for r in rows})}")
    print(f"  唯一 (script,char): {len({(r.get('script',''), r.get('character','')) for r in rows})}")
    print(f"  唯一 glyph_id   : {len({r.get('glyph_id','') for r in rows})}")
    print(f"  书体分布        : {dict(Counter(r.get('script','') for r in rows))}")
    print(f"  按源            : {dict(Counter(r['image_path'].split('/')[2] for r in rows))}")
    cc = Counter(r.get("character", "") for r in rows)
    v = sorted(cc.values())
    print(f"  每字样本数      : min={v[0]} p25={v[len(v)//4]} p50={v[len(v)//2]} "
          f"p75={v[3*len(v)//4]} max={v[-1]}")

    print(f"\n  书法家明细 (Top20):")
    for k, n in Counter(r.get("calligrapher", "") for r in rows).most_common(20):
        print(f"    {k:12s} {n}")

    print(f"\n=== 60px 判据 (w_max > {TH:.0f}px) ===", flush=True)
    with mp.Pool(40) as pool:
        w = pool.map(probe, [r["image_path"] for r in rows], chunksize=256)
    w = np.array(w)
    kill = w > TH
    print(f"  会剔掉 {int(kill.sum())} / {len(rows)} ({100*kill.mean():.2f}%)")
    print(f"  剩余   {int((~kill).sum())}")
    for s in KEEP:
        idx = [i for i, r in enumerate(rows) if s in r["image_path"]]
        k = int(kill[idx].sum())
        print(f"    {s:22s} {len(idx):6d} -> 剔 {k:5d} ({100*k/len(idx):5.2f}%) "
              f"剩 {len(idx)-k}")
    print(f"\n  w_max 分布: p50={np.percentile(w,50):.1f} "
          f"p75={np.percentile(w,75):.1f} p90={np.percentile(w,90):.1f} "
          f"p99={np.percentile(w,99):.1f} max={w.max():.1f}")
    for t in (40, 50, 60, 80, 100):
        print(f"    w>{t:3.0f}: 剔 {int((w>t).sum()):6d}")


if __name__ == "__main__":
    main()
