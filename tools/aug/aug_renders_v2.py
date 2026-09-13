# -*- coding: utf-8 -*-
"""
aug_renders_v2.py — 安全墨迹增强 (骨架 g 条件不动, 不做宽度双向扰动).

每图 2 个变体:
  _a: 墨密度温和扰动 (×0.8-1.25, 风格不变 — 同一人换墨浓淡)
  _b: 去噪 + 选择性加粗 (底噪钳制; 仅当笔画过细/易断时 +1px 补粗, 治断笔, 不动粗风格)
变体 _c (shift/rotate ±4px/±1.5°) 需与骨架 g 联动, 记录变换参数, encode 阶段同步套用到骨架.
输出: assets/aug_renders_fame_v2/<id>_a.png / _b.png
      assets/aug_transforms_v2.json (仅 _c 的变换参数)
      assets/train_fame3_aug_v2.csv
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
OUT_DIR = f"{ROOT}/assets/aug_renders_fame_v2"
OUT_CSV = f"{ROOT}/assets/train_fame3_aug_v2.csv"
OUT_TF = f"{ROOT}/assets/aug_transforms_v2.json"
sys.stdout.reconfigure(encoding="utf-8")

os.makedirs(OUT_DIR, exist_ok=True)
rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))


def ink_domain(im):
    """-> (ink float arr: 墨=正, 背景=0, 已去底噪)"""
    arr = 255.0 - np.asarray(im, dtype=np.float32)
    arr[arr < 8] = 0.0          # 底噪钳制: 扫描灰点/纸纹 -> 纯白
    return arr


def stroke_width(arr):
    """笔画中位宽度(px): 2×到背景的距离中位数; 无墨返回 0"""
    try:
        import scipy.ndimage as ndi
    except ImportError:
        mask = arr > 40
        return 2 * np.sqrt((mask).sum() / max(1, (arr > 127).sum())) if (arr > 127).sum() else 0.0
    mask = arr > 40
    if mask.sum() < 20:
        return 0.0
    d = ndi.distance_transform_edt(mask)
    return float(np.median(d[mask])) * 2.0


def make_variant(im, rng, kind):
    """kind: a=墨密度, b=去噪+选择性加粗, c=shift/rotate(仅墨图; 骨架同步变换由 encode 阶段做)"""
    arr = ink_domain(im)
    info = {}
    if kind == "a":
        arr = np.clip(arr * rng.uniform(0.8, 1.25), 0, 255)
    elif kind == "b":
        w = stroke_width(arr)
        im2 = Image.fromarray((255.0 - arr).astype(np.uint8))
        passes = 0
        if 0 < w < 1.4:
            passes = 2
        elif w < 2.2:
            passes = 1
        for _ in range(passes):
            im2 = im2.filter(ImageFilter.MinFilter(3))  # 仅对细字补粗
        arr = ink_domain(im2)
        info["width"] = round(w, 2)
        info["thicken_passes"] = passes
    else:  # c
        ang = rng.uniform(-1.5, 1.5)
        dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
        try:
            import scipy.ndimage as ndi
            arr = ndi.rotate(arr, ang, reshape=False, order=1, mode="constant", cval=0.0)
            arr = ndi.shift(arr, (dy, dx), order=1, mode="constant", cval=0.0)
        except ImportError:
            im2 = Image.fromarray((255.0 - arr).astype(np.uint8)).rotate(ang, resample=Image.BILINEAR, fillcolor=255)
            arr = 255.0 - np.asarray(im2, dtype=np.float32)
        info["angle"] = round(ang, 3)
        info["shift"] = [dx, dy]
    im2 = Image.fromarray((255.0 - np.clip(arr, 0, 255)).astype(np.uint8))
    return im2, info


def work(arg):
    idx, r = arg
    rng = random.Random(10_000 + idx)
    src = os.path.join(ROOT, r["image_path"])
    try:
        im = Image.open(src).convert("L")
    except Exception as e:
        return (idx, f"FAIL {src}: {e}", None)
    base = os.path.splitext(os.path.basename(r["image_path"]))[0]
    out, tf = {}, {}
    for kind in ("a", "b", "c"):
        p = os.path.join(OUT_DIR, f"{base}_{kind}.png")
        v, info = make_variant(im, rng, kind)
        if kind == "c":
            if info["shift"] != [0, 0] or info["angle"] != 0.0:
                tf[f"{base}_c"] = info
        if not os.path.exists(p):
            v.save(p)
        out[kind] = str(info or "")
    return (idx, out, tf)


if __name__ == "__main__":
    tasks = [(i, r) for i, r in enumerate(rows)]
    done, tfs = {}, {}
    with Pool(48) as pool:
        for n, (idx, out, tf) in enumerate(pool.imap_unordered(work, tasks, chunksize=64)):
            done[idx] = out
            if tf:
                tfs.update(tf)
            if (n + 1) % 5000 == 0:
                print(f"{n + 1}/{len(tasks)}")
    fails = [k for k, v in done.items() if isinstance(v, str)]
    print("done:", len(done) - len(fails), "fails:", len(fails))
    if fails:
        for k in fails[:5]:
            print(" ", done[k])
    json.dump(tfs, open(OUT_TF, "w", encoding="utf-8"))
    print(f"{OUT_TF}: {len(tfs)} shift/rotate 变换记录 (encode 阶段骨架同步用)")

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["aug"])
        w.writeheader()
        for i, r in enumerate(rows):
            out = done[i]
            if isinstance(out, str):
                continue
            base = os.path.splitext(os.path.basename(r["image_path"]))[0]
            for kind in ("a", "b", "c"):
                r2 = dict(r)
                r2["image_path"] = f"assets/aug_renders_fame_v2/{base}_{kind}.png"
                r2["aug"] = f"{kind}{out[kind] and ':' + out[kind]}"
                w.writerow(r2)
        for i, r in enumerate(rows):
            r2 = dict(r)
            r2["aug"] = "orig"
            w.writerow(r2)
    n = sum(1 for _ in open(OUT_CSV, encoding="utf-8")) - 1
    print(f"{OUT_CSV}: {n} rows (期望 28385×(1+3) = 113540)")
