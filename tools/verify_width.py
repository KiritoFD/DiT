# -*- coding: utf-8 -*-
"""verify_width.py — 交叉验证"单笔画粗度"度量是否与视觉一致.

两种独立度量:
  A) w_max = distance_transform_edt(mask).max() * 2      (最大内切圆直径)
  B) w_ero = 2 * (mask 被完全腐蚀掉所需的迭代次数)         (形态学口径)
  C) w_avg = 墨面积 / 骨架长度                            (平均笔画宽)
三者对一个规则矩形笔画应一致。若 A 明显小于视觉粗度, 说明 mask 不完整
(笔画内部有灰阶) -> 应改用更宽松的阈值 (a<某值) 或按亮度加权。

用法: python tools/verify_width.py --csv assets/train_base_clean.csv --n 12
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label, generate_binary_structure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ST = generate_binary_structure(2, 2)


def measures(a, thr):
    m = a < thr
    if not m.any():
        return 0.0, 0.0, 0.0, 0.0
    w_dt = float(distance_transform_edt(m).max()) * 2.0
    mm = m.copy()
    i = 0
    while mm.any() and i < 300:
        mm = binary_erosion(mm, ST)
        i += 1
    w_ero = float(i) * 2.0
    lab, _ = label(m)
    sizes = np.bincount(lab.ravel())[1:]
    blob = float(sizes.max()) / a.size
    return w_dt, w_ero, blob, float(m.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_base_clean.csv")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sort", default="wdt_desc")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"[verify] {a.csv}: {len(rows)} 行")
    # 只看单字笔画简单、便于目测的字
    tgt = [r for r in rows if r.get("character") in ("一", "七", "二", "十", "三")]
    print(f"  取 一/七/二/十/三 共 {len(tgt)} 张")

    recs = []
    for r in tgt[:3000]:
        p = r["image_path"]
        if not os.path.exists(p):
            continue
        arr = np.asarray(Image.open(p).convert("L"))
        for thr in (128, 160, 190):
            pass
        w_dt, w_ero, blob, ink = measures(arr, 128)
        w_dt2, _, _, _ = measures(arr, 190)
        recs.append((p, r.get("character", ""), r.get("calligrapher", ""),
                     w_dt, w_ero, w_dt2, blob, ink))
    recs.sort(key=lambda x: -x[3])
    print(f"\n  {'img':44s} {'字':2s} {'w_dt':>6s} {'w_ero':>6s} {'w_dt190':>7s} "
          f"{'blob':>6s} {'ink':>6s}")
    for p, ch, cal, w1, w2, w3, b, ik in recs[:a.n]:
        print(f"  {p[-42:]:44s} {ch:2s} {w1:6.1f} {w2:6.1f} {w3:7.1f} {b:6.3f} {ik:6.3f}")

    os.makedirs("/tmp/wv", exist_ok=True)
    for i, (p, ch, cal, w1, w2, w3, b, ik) in enumerate(recs[:8]):
        shutil_ok = True
        try:
            im = Image.open(p).convert("L")
            im.resize((256, 256), Image.NEAREST).save(f"/tmp/wv/{i}_wdt{w1:.0f}_{ch}.png")
        except Exception:
            shutil_ok = False
    print(f"\n  -> /tmp/wv (原尺寸 256, NEAREST 放大便于目测笔画宽度)")

    # 全局分布 (对 一/七 这类简单字)
    ws = np.array([x[3] for x in recs])
    if ws.size:
        print(f"\n  简单字(一/七/二/十/三) w_dt 分布: p50={np.percentile(ws,50):.1f} "
              f"p75={np.percentile(ws,75):.1f} p90={np.percentile(ws,90):.1f} "
              f"max={ws.max():.1f}  n={ws.size}")


if __name__ == "__main__":
    main()
