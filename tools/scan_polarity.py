# -*- coding: utf-8 -*-
"""
scan_polarity.py — 反色/极性筛查: 全库按行扫图, 判定 背景极性.

判定 (256 灰度图):
  border_mean = 图像边缘 10px 框的均值 (背景亮度代理)
  ink_ratio   = 像素 < 阈值 的占比 (墨迹代理)
  极性正常: border_mean > 170 (白底) 且 ink_ratio < 0.55
  反色嫌疑: border_mean < 100 (黑底拓片) 或 ink_ratio > 0.55
  低对比:  170 >= border_mean >= 100 (背景灰, 需目检)

输出: base_polarity_report.txt (按来源/书家统计) + base_polarity_bad.csv (嫌疑行)
"""
import collections as C
import csv
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")


def one(task):
    iid, path, src, callig = task
    try:
        im = Image.open(path).convert("L")
        if im.size != (256, 256):
            im = im.resize((256, 256))
        a = np.asarray(im, dtype=np.float32)
        b = 10
        border = np.concatenate([a[:b].ravel(), a[-b:].ravel(),
                                 a[:, :b].ravel(), a[:, -b:].ravel()])
        border_mean = float(border.mean())
        ink_ratio = float((a < 128).mean())
        center_mean = float(a[64:192, 64:192].mean())
        verdict = "ok"
        if border_mean < 100 or ink_ratio > 0.55:
            verdict = "inverted"
        elif border_mean < 170:
            verdict = "low_contrast"
        return iid, src, callig, verdict, border_mean, ink_ratio, center_mean
    except Exception as e:
        return iid, src, callig, f"error:{e}", -1, -1, -1


def main():
    rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
    tasks = []
    for r in rows:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        src = "unicalli" if "unicalli" in r["image_path"] else (
            "tongji" if "tongji" in r["image_path"] else "fame")
        tasks.append((iid, r["image_path"], src, r["calligrapher"]))
    print(f"scanning {len(tasks)} images...", flush=True)
    t0 = time.time()
    results = []
    with mp.Pool(48) as pool:
        for k, res in enumerate(pool.imap_unordered(one, tasks, chunksize=64), 1):
            results.append(res)
            if k % 10000 == 0:
                print(f"  {k}/{len(tasks)} ({k/(time.time()-t0):.0f}/s)", flush=True)
    print(f"scanned in {time.time()-t0:.0f}s", flush=True)

    by_src = C.Counter()
    by_src_verdict = C.defaultdict(C.Counter)
    by_callig_bad = C.Counter()
    by_callig_total = C.Counter()
    for iid, src, callig, verdict, bm, ir, cm in results:
        by_src[src] += 1
        by_src_verdict[src][verdict] += 1
        by_callig_total[callig] += 1
        if verdict in ("inverted", "low_contrast", "error") or verdict.startswith("error"):
            by_callig_bad[callig] += 1

    with open("base_polarity_bad.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "src", "calligrapher", "verdict", "border_mean",
                    "ink_ratio", "center_mean"])
        for iid, src, callig, verdict, bm, ir, cm in results:
            if verdict != "ok":
                w.writerow([iid, src, callig, verdict, f"{bm:.1f}", f"{ir:.3f}", f"{cm:.1f}"])

    out = open("base_polarity_report.txt", "w", encoding="utf-8")
    out.write(f"total scanned: {len(results)}\n\n")
    for src in ("fame", "tongji", "unicalli"):
        v = by_src_verdict.get(src, {})
        out.write(f"[{src}] {by_src[src]}: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(v.items())) + "\n")
    out.write("\n书家 (嫌疑图数/总数, 只列有嫌疑的):\n")
    for c, bad in by_callig_bad.most_common():
        out.write(f"  {c}: {bad}/{by_callig_total[c]}\n")
    out.close()
    print("written base_polarity_report.txt + base_polarity_bad.csv", flush=True)


if __name__ == "__main__":
    main()
