#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_mid_carriers.py — 纯 CPU 预建中程结构 loss 的两种载体 PNG。

对应讨论: 中程结构 loss 的 target 不能是细骨架, 要粗化到 x0_pred 的天然形态。
本脚本只做**图片准备**(CPU), 不碰 VAE encode / 不占 GPU。

两种载体:
  A) 多宽度骨架  skel_w{w}.png
     std 骨架 PNG -> 二值(ink<128) -> skeletonize 到 1px -> binary_dilation[(w-1)//2]
     实测等效笔宽 2.98/4.95/6.91/8.85/10.79 px 对应 w=3/5/7/9/11
     ★ 必须 skeletonize: 原 std 图已有 ~3.5px 笔宽, 直接 dilate 会偏大 ~2.5px
  B) 高斯模糊 GT  gt_blur_s{sigma}.png
     GT 图像 PNG -> 高斯模糊 σ px
     供 blur_gt 载体按 σ(t) 分档取用

输出结构:
  <out_root>/w3/<img_id>.png       多宽度骨架
  <out_root>/w5/<img_id>.png
  ...
  <out_root>/blur/s050/<img_id>.png  高斯模糊 GT (σ=0.5 ... )
  <out_root>/manifest.json           记录参数与统计

用法 (远端, 多进程 CPU):
  python tools/prepare_mid_carriers.py --csv assets/train_50k_v2.csv \\
      --widths 3,5,7,9,11 --sigmas 0.5,1,1.5,2,2.5,3,4 \\
      --out-root data/50k/mid_carriers --nproc 32
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# 全局（子进程继承）
_G = {}


def _init(widths, sigmas, out_root, size):
    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize
    _G["widths"] = widths
    _G["sigmas"] = sigmas
    _G["out_root"] = out_root
    _G["size"] = size
    _G["binary_dilation"] = binary_dilation
    _G["skeletonize"] = skeletonize
    for w in widths:
        os.makedirs(os.path.join(out_root, f"w{w}"), exist_ok=True)
    for s in sigmas:
        os.makedirs(os.path.join(out_root, "blur", _stag(s)), exist_ok=True)


def _stag(sigma):
    """σ 目录名: 1.0 -> 's1'  1.5 -> 's1p5'  2.0 -> 's2'  0.5 -> 's0p5'
    ★ 不能用 replace('.','') —— 1.0->'s10' 会与 10.0 混淆, 1.5->'s15' 与 15.0 混淆。
    """
    s = ("%g" % float(sigma))
    return "s" + s.replace(".", "p").replace("-", "m")


def _prep_one(item):
    """处理一条: 生成所有宽度的骨架 PNG + 所有 σ 的模糊 GT PNG。"""
    img_id, std_path, img_path = item
    bd = _G["binary_dilation"]
    sk_fn = _G["skeletonize"]
    out = _G["out_root"]
    size = _G["size"]
    made = 0
    err = ""

    # ---- A) 多宽度骨架 ----
    if std_path and os.path.exists(std_path):
        try:
            g = np.asarray(Image.open(std_path).convert("L"))
            ink = g < 128
            if ink.sum() > 0:
                sk = sk_fn(ink)
            else:
                sk = ink
            for w in _G["widths"]:
                it = max(0, (w - 1) // 2)
                d = bd(sk, iterations=it) if it > 0 else sk
                arr = np.where(d, 0, 255).astype(np.uint8)   # ink=黑
                Image.fromarray(arr).save(os.path.join(out, f"w{w}", f"{img_id}.png"))
                made += 1
        except Exception as e:
            err += f"skel:{e};"
    else:
        err += "std_missing;"

    # ---- B) 高斯模糊 GT ----
    if img_path and os.path.exists(img_path):
        try:
            from PIL import ImageFilter
            im = Image.open(img_path).convert("L")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BILINEAR)
            for s in _G["sigmas"]:
                # PIL GaussianBlur 的 radius ≈ σ (足够近似; 精确版可用 scipy)
                b = im.filter(ImageFilter.GaussianBlur(radius=float(s))) if s > 0 else im
                b.save(os.path.join(out, "blur", _stag(s), f"{img_id}.png"))
                made += 1
        except Exception as e:
            err += f"blur:{e};"
    else:
        err += "img_missing;"

    return (img_id, made, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_50k_v2.csv")
    ap.add_argument("--std-col", default="std_path")
    ap.add_argument("--img-col", default="image_path")
    ap.add_argument("--widths", default="3,5,7,9,11")
    ap.add_argument("--sigmas", default="0.5,1,1.5,2,2.5,3,4")
    ap.add_argument("--out-root", default="data/50k/mid_carriers")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="调试: 只做前 N 条")
    a = ap.parse_args()

    widths = [int(x) for x in a.widths.split(",") if x.strip()]
    sigmas = [float(x) for x in a.sigmas.split(",") if x.strip()]

    import csv as _csv
    rows = list(_csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]

    def img_id(r):
        for c in ("img_id", "old_50k_id"):
            if str(r.get(c, "")).strip():
                return str(r[c]).zfill(6) if str(r[c]).isdigit() else str(r[c])
        return os.path.splitext(os.path.basename(r.get(a.img_col, "") or "x"))[0]

    items = [(img_id(r), (r.get(a.std_col, "") or "").strip(),
              (r.get(a.img_col, "") or "").strip()) for r in rows]

    print("=" * 62)
    print(f" 中程载体图片准备 (纯 CPU)")
    print("=" * 62)
    print(f" CSV: {a.csv}  ({len(items)} 条)")
    print(f" 骨架宽度: {widths}  -> {a.out_root}/w{{w}}/")
    print(f" 模糊 σ:   {sigmas}  -> {a.out_root}/blur/s*/")
    print(f" 进程数: {a.nproc}  输出: {a.out_root}")
    print("-" * 62)

    t0 = time.time()
    n_ok = 0
    n_made = 0
    errs = {}
    with Pool(a.nproc, initializer=_init,
              initargs=(widths, sigmas, a.out_root, a.size)) as pool:
        for i, (iid, made, err) in enumerate(
                pool.imap_unordered(_prep_one, items, chunksize=16), 1):
            n_made += made
            if err:
                for e in err.split(";"):
                    if e:
                        errs[e] = errs.get(e, 0) + 1
            else:
                n_ok += 1
            if i % 2000 == 0 or i == len(items):
                el = time.time() - t0
                eta = el / i * (len(items) - i)
                print(f"  [{i}/{len(items)}] {el:.0f}s  ETA {eta:.0f}s  "
                      f"产出 {n_made} 张")
    dt = time.time() - t0

    print("-" * 62)
    print(f" 完成: {n_ok} 条无错 / {len(items)} 条  |  共 {n_made} 张 PNG")
    if errs:
        print(f" 错误统计: {errs}")
    print(f" 用时: {dt:.1f}s")

    man = {
        "csv": a.csv, "n_rows": len(items), "widths": widths, "sigmas": sigmas,
        "size": a.size, "out_root": a.out_root, "n_png": n_made,
        "seconds": round(dt, 1), "errors": errs,
        "note": "骨架=skeletonize后binary_dilation[(w-1)//2]; 模糊=PIL GaussianBlur(radius=σ)",
    }
    os.makedirs(a.out_root, exist_ok=True)
    json.dump(man, open(os.path.join(a.out_root, "manifest.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f" manifest -> {a.out_root}/manifest.json")
    print("=" * 62)


if __name__ == "__main__":
    main()
