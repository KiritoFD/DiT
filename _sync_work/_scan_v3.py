# -*- coding: utf-8 -*-
"""_scan_v3.py — 黑底图最终统计 (判据: 墨占比 ink).

判据依据 (实测校准):
  正常白底书法 ink=(a<128).mean() ∈ [0.03, 0.45]   (fame 27,552 张实测 max=0.381)
  黑底/拓片/死黑   ink > 0.5                        (背景是墨)
  -> 反相后前景 bright = 1 - ink
       bright < 0.02          -> dead_black  (几乎无字形, 只能删)
       0.02 <= bright <= 0.55 -> invertible  (反相可救)
       bright > 0.55          -> odd         (边界, 人工看)
输出 montage 供肉眼验证。
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
MONT = f"/root/Workspace/xy/DiT/_otout_v3_{TAG}"
os.makedirs(MONT, exist_ok=True)

INK_HI = 0.50      # 墨占比高于此 -> 底色异常
DEAD = 0.02        # 反相后前景低于此 -> 无内容


def probe(r):
    p = r.get("image_path", "")
    meta = (r.get("character", ""), r.get("calligrapher", ""), r.get("script", ""),
            r.get("aug", "") or "-")
    try:
        a = np.asarray(Image.open(p).convert("L"))
    except Exception:
        return (p, *meta, 0.0, 0.0, 0.0, "read_error")
    ink = float((a < 128).mean())
    bright = 1.0 - ink
    med = float(np.median(a))
    if ink <= INK_HI:
        v = "ok"
    elif bright < DEAD:
        v = "dead_black"
    elif bright <= 0.55:
        v = "invertible"
    else:
        v = "odd"
    return (p, *meta, ink, bright, med, v)


def montage(paths, out):
    CELL, COLS = 96, 8
    n = min(len(paths), 64)
    nrow = (n + COLS - 1) // COLS
    cv = Image.new("RGB", (COLS * CELL, max(nrow, 1) * CELL), (128, 0, 0))
    for i, p in enumerate(paths[:n]):
        try:
            im = Image.open(p).convert("L").resize((CELL, CELL))
            cv.paste(im.convert("RGB"), ((i % COLS) * CELL, (i // COLS) * CELL))
        except Exception:
            pass
    cv.save(out)


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    print(f"[scan] {CSV}: {len(rows)} rows", flush=True)
    t0 = time.time()
    res = []
    with mp.Pool(40) as pool:
        for n, rr in enumerate(pool.imap_unordered(probe, rows, chunksize=256), 1):
            res.append(rr)
            if n % 40000 == 0:
                print(f"  {n}/{len(rows)} ({n/(time.time()-t0):.0f}/s)", flush=True)

    cnt = Counter(r[-1] for r in res)
    tot = len(res)
    print(f"\n=== {TAG} 结果 ({time.time()-t0:.0f}s, {tot} 行) ===")
    for k in ("ok", "invertible", "dead_black", "odd", "read_error"):
        c = cnt.get(k, 0)
        print(f"  {k:12s} {c:7d}  ({100*c/tot:6.2f}%)")
    bad = cnt.get("invertible", 0) + cnt.get("dead_black", 0) + cnt.get("odd", 0)
    print(f"  {'黑底合计':12s} {bad:7d}  ({100*bad/tot:6.2f}%)")

    def src_of(p):
        parts = p.split("/")
        return parts[2] if len(parts) > 2 else "?"
    by, uniq = {}, {"invertible": set(), "dead_black": set(), "odd": set()}
    for r in res:
        s = src_of(r[0])
        by.setdefault(s, Counter())[r[-1]] += 1
        if r[-1] in uniq:
            uniq[r[-1]].add(r[0])
    print("\n  按数据源 (行数):")
    for s, d in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        print(f"    {s:22s} 总={sum(d.values()):7d} "
              f"可救={d.get('invertible',0):6d} 死黑={d.get('dead_black',0):5d} "
              f"odd={d.get('odd',0):4d}")
    print("\n  唯一图: " + "  ".join(f"{k}={len(v)}" for k, v in uniq.items()))

    # 墨占比分布 (ok 部分)
    oks = np.array([r[5] for r in res if r[-1] == "ok"])
    if oks.size:
        print(f"\n  ok 图 ink 分布: p50={np.percentile(oks,50):.3f} "
              f"p99={np.percentile(oks,99):.3f} max={oks.max():.3f}")

    # 关键已知图核对
    KEY = ["960697", "960706", "970622", "960132", "960760", "976891", "960761"]
    print(f"\n  关键图核对:")
    for r in res:
        b = os.path.basename(r[0])[:-4]
        if b in KEY:
            print(f"    {b} ink={r[5]:.3f} bright={r[6]:.3f} med={r[7]:.0f} "
                  f"-> {r[8]}  char={r[1]} {r[2]}")

    print(f"\n  死黑样例:")
    for r in [x for x in res if x[-1] == "dead_black"][:10]:
        print(f"    {r[0]} ink={r[5]:.3f} bright={r[6]:.4f} med={r[7]:.0f} "
              f"char={r[1]} {r[2]}")
    print(f"\n  可救样例:")
    for r in [x for x in res if x[-1] == "invertible"][:10]:
        print(f"    {r[0]} ink={r[5]:.3f} bright={r[6]:.3f} med={r[7]:.0f} "
              f"char={r[1]} {r[2]}")

    # montage
    for nm, k in (("ok", "ok"), ("inv", "invertible"), ("dead", "dead_black")):
        ps = [x[0] for x in res if x[-1] == k][:64]
        montage(ps, f"{MONT}/{nm}.png")
    print(f"  -> montage: {MONT} (ok.png / inv.png / dead.png)")

    with open(f"/tmp/v3_{TAG}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "character", "calligrapher", "script", "aug",
                    "ink", "bright_after_invert", "median", "verdict"])
        for r in res:
            if r[-1] != "ok":
                w.writerow(r)
    print(f"  -> /tmp/v3_{TAG}.csv")


if __name__ == "__main__":
    main()
