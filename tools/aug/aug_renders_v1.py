# -*- coding: utf-8 -*-
"""
aug_renders_v1.py — 墨迹渲染扰动 (target-only, 骨架 g 条件不动).

每个训练图生成 2 个变体:
  _a: 墨密度扰动  (gamma 0.75-1.3 + contrast 0.85-1.15 + 可选轻度模糊 σ≤0.8)
  _b: 笔画宽度扰动 (±1px 膨胀/腐蚀) + 轻度 gamma 抖动
骨架条件 g 不变 → 同一 (g, callig) 条件对应 K 个不同目标 → 打击"背单一图"记忆化.
输出: 5script/aug_renders_fame_v1/<id>_a.png / _b.png
      5script/train_fame3_aug_v1.csv (原 28385 行 + 56770 增强行)
"""
import csv
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/5script/train_fame3_clean_v8.csv"
OUT_DIR = f"{ROOT}/5script/aug_renders_fame_v1"
OUT_CSV = f"{ROOT}/5script/train_fame3_aug_v1.csv"
sys.stdout.reconfigure(encoding="utf-8")

os.makedirs(OUT_DIR, exist_ok=True)
rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))


def gamma_lut(g):
    return [min(255, int(255.0 * ((i / 255.0) ** g))) for i in range(256)]


def make_variant(im, rng, kind):
    arr = 255.0 - np.asarray(im, dtype=np.float32)  # 墨量域: 背景=0, 只缩放墨
    if kind == "a":
        arr = np.clip(arr * rng.uniform(0.65, 1.35), 0, 255)  # 墨密度
        im = Image.fromarray((255.0 - arr).astype(np.uint8))
        r = rng.random()
        if r < 0.35:
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    else:
        if rng.random() < 0.5:
            im = im.filter(ImageFilter.MinFilter(3))   # 墨变粗 (背景白, 取 min 扩墨)
        else:
            im = im.filter(ImageFilter.MaxFilter(3))   # 墨变细
        arr = 255.0 - np.asarray(im, dtype=np.float32)
        arr = np.clip(arr * rng.uniform(0.85, 1.15), 0, 255)
        im = Image.fromarray((255.0 - arr).astype(np.uint8))
    return im


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
    for kind in ("a", "b"):
        p = os.path.join(OUT_DIR, f"{base}_{kind}.png")
        if not os.path.exists(p):
            make_variant(im, rng, kind).save(p)
        out[kind] = p
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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for i, r in enumerate(rows):
            out = done[i]
            if isinstance(out, str):
                continue
            base = os.path.splitext(os.path.basename(r["image_path"]))[0]
            for kind in ("a", "b"):
                r2 = dict(r)
                r2["image_path"] = f"5script/aug_renders_fame_v1/{base}_{kind}.png"
                r2["aug"] = f"ink_{kind}"
                w.writerow(r2)
        # 原始行也重写进 aug csv (统一入口), aug 列 = orig
        for i, r in enumerate(rows):
            r2 = dict(r)
            r2["aug"] = "orig"
            w.writerow(r2)
    n = sum(1 for _ in open(OUT_CSV, encoding="utf-8")) - 1
    print(f"{OUT_CSV}: {n} rows (期望 28385 + 成功变体数)")
