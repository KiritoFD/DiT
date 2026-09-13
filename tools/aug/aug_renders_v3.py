# -*- coding: utf-8 -*-
"""
aug_renders_v3.py — 墨迹增强 v3.3: 二值笔画加粗/收窄, 三档强度.

b1/b2/b3 = 同一算子不同强度: 测笔画宽 w, 朝带中 mid=(4.5+9)/2=6.75 收敛:
  target = K*(mid-w), 封顶 cap_frac*w, 每 pass ≈ ±2px, 离散二值形态学(纯黑白无中间灰)
  b1: K=0.6 cap 0.35w   b2: K=1.2 cap 0.5w   b3: K=2.0 cap 0.65w
细→粗 / 粗→细, 更好看更好学. 无 _a/_c.
输出: data/imgs/fame3-e/<id>_{b1,b2,b3}.png + assets/train_fame3_e.csv
"""
import csv
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFilter

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/assets/train_fame3_clean_v8.csv"
OUT_DIR = f"{ROOT}/data/imgs/fame3-e"
OUT_CSV = f"{ROOT}/assets/train_fame3_e.csv"
sys.stdout.reconfigure(encoding="utf-8")

LO, HI = 4.5, 9.0
MID = (LO + HI) / 2.0
STRENGTHS = {"b1": (0.6, 0.35), "b2": (1.2, 0.50), "b3": (2.0, 0.65)}  # (K, cap_frac)
KINDS = ("b1", "b2", "b3")

os.makedirs(OUT_DIR, exist_ok=True)
rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))


def ink_domain(im):
    arr = 255.0 - np.asarray(im, dtype=np.float32)
    arr[arr < 8] = 0.0  # 底噪钳制
    return arr


def stroke_width(arr):
    import scipy.ndimage as ndi
    mask = arr > 40
    if mask.sum() < 20:
        return 0.0
    return float(np.median(ndi.distance_transform_edt(mask)[mask])) * 2.0


def make_variant(im, rng, kind):
    K, capf = STRENGTHS[kind]
    arr = ink_domain(im)
    info = {}
    w = stroke_width(arr)
    info["w"] = round(w, 2)
    if w <= 0:
        info["delta"] = 0
        return Image.fromarray((255.0 - arr).astype(np.uint8)), info
    target = float(np.clip(K * (MID - w), -capf * w, capf * w))
    passes = int(round(target / 2.0))
    if passes == 0 and abs(target) > 0.2:
        passes = int(np.sign(target))
    imc = Image.fromarray((255.0 - arr).astype(np.uint8))
    for _ in range(abs(passes)):  # 每 pass ≈ ±2px, 离散二值形态学
        imc = imc.filter(ImageFilter.MinFilter(3) if passes > 0 else ImageFilter.MaxFilter(3))
    arr = ink_domain(imc)
    arr = (arr > 127) * 255.0  # 二值化: 纯黑白
    info["delta"] = passes * 2
    im2 = Image.fromarray((255.0 - arr).astype(np.uint8))
    return im2, info


def work(arg):
    idx, r = arg
    rng = random.Random(10_000 + idx)
    src = os.path.join(ROOT, r["image_path"])
    try:
        im = Image.open(src).convert("L")
    except Exception as e:
        return (idx, f"FAIL {src}: {e}")
    base = os.path.splitext(os.path.basename(r["image_path"]))[0]
    out = {}
    for kind in KINDS:
        p = os.path.join(OUT_DIR, f"{base}_{kind}.png")
        if not os.path.exists(p):
            make_variant(im, rng, kind)[0].save(p)
        out[kind] = "ok"
    return (idx, out)


if __name__ == "__main__":
    tasks = [(i, r) for i, r in enumerate(rows)]
    done = {}
    with Pool(48) as pool:
        for n, (idx, out) in enumerate(pool.imap_unordered(work, tasks, chunksize=64)):
            done[idx] = out
            if (n + 1) % 5000 == 0:
                print(f"{n + 1}/{len(tasks)}")
    fails = [k for k, v in done.items() if isinstance(v, str)]
    print("done:", len(done) - len(fails), "fails:", len(fails))
    if fails:
        for k in fails[:5]:
            print(" ", done[k])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["aug"])
        w.writeheader()
        for i, r in enumerate(rows):
            if isinstance(done[i], str):
                continue
            base = os.path.splitext(os.path.basename(r["image_path"]))[0]
            for kind in KINDS:
                r2 = dict(r)
                r2["image_path"] = f"data/imgs/fame3-e/{base}_{kind}.png"
                r2["aug"] = kind
                w.writerow(r2)
    n = sum(1 for _ in open(OUT_CSV, encoding="utf-8")) - 1
    print(f"{OUT_CSV}: {n} rows")
