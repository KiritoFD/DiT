#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_aug_geo.py — 书法专属数据增强构建器 v2（文献四方案中的 ①②③）

与 build_aug_variants.py 的关系
------------------------------
旧脚本只做**二值形态学**（8 邻域/十字），因为"笔宽变化不改变 DINO patch-token →
REPA 特征缓存可复用"。本脚本补齐文献里的三族增强，并且**明确承担 REPA 代价**：

  ☠ 凡改变**几何**的增强（弹性/网格/仿射）都会改变 DINO 特征，
     **增强行的 REPA 特征必须重算**（tools/ablation/expand_repa_cache.py）。

三族实现（零新依赖: 只用 PIL + scipy，48/4090 都装好了）
------------------------------------------------------
  morph   : ① 形态学粗细抖动 —— **椭圆结构元**（对齐 cv2.MORPH_ELLIPSE）+ 每样本随机概率
  elastic : ② 弹性形变  —— 高斯平滑随机位移场（Simard et al. 经典做法）
  grid    : ② 网格形变  —— 4×4 控制点抖动后双线性插值成位移场
  affine  : ③ 限制性仿射 —— **禁止旋转**；只做 ±3° 错切 / 0.95~1.05 缩放 / ±2% 平移

所有几何变换都在 **墨迹通道**（ink = 255 - gray）上做重采样再反相 ——
这样插值是在"无墨 = 0"一侧混合，不会把纸面变成灰底。

保护规则（防止字变乱码，与旧脚本一致）
--------------------------------------
  加粗: 面积 ≤ 3× 原面积;  变细: 面积 ≥ 15% 且 ≥ 20 px
  几何: 墨迹面积变化 ≤ 35%，且不得出现全白/全黑

用法
----
  # 自检（生成对比拼图, 不写数据集）
  python tools/ablation/build_aug_geo.py --selftest --out-png scratch/aug_geo_selftest.png

  # 批量生成（与 build_aug_variants.py 同风格: 图目录 + 全景 CSV + manifest）
  python tools/ablation/build_aug_geo.py --strategy geo3x --root . \
      --src-csv exp-std/csv/train.csv \
      --out-csv exp-std/csv/train_top10_aug_geo3x.csv \
      --imgs-dir data/top10_style23/imgs_aug_geo3x --workers 32
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation, binary_erosion, gaussian_filter, map_coordinates,
)

# ---------------------------------------------------------------- 结构元 / 保护
THIN_MIN_FRAC, THIN_MIN_PX = 0.15, 20          # 变细保护
THICK_MAX_MULT = 3.0                           # 加粗保护
GEO_MAX_AREA_DELTA = 0.35                      # 几何: 面积变化上限


def ellipse_struct(r):
    """半径 r 的椭圆结构元（等价 cv2.getStructuringElement(MORPH_ELLIPSE)）。"""
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2 + 1e-9


def _ink(gray):
    """灰度 -> 墨迹(0..1, 1=墨)。约定: 白纸黑墨。"""
    return 1.0 - gray.astype(np.float32) / 255.0


def _to_gray(ink):
    return np.clip((1.0 - ink) * 255.0, 0, 255).astype(np.uint8)


def _area_ok(ink0, ink1, mode):
    a0 = float((ink0 > 0.5).sum())
    a1 = float((ink1 > 0.5).sum())
    if a0 < 1 or a1 < 1:
        return False
    if mode == "morph_thick":
        return a1 <= THICK_MAX_MULT * a0
    if mode == "morph_thin":
        return a1 >= THIN_MIN_FRAC * a0 and a1 >= THIN_MIN_PX
    return abs(a1 - a0) / max(a0, 1.0) <= GEO_MAX_AREA_DELTA


# ---------------------------------------------------------------- 各增强实现
def aug_morph(ink, rng, p=1.0):
    """① 形态学粗细抖动: 椭圆核 dilate/erode, 核半径 1~2 (≈3x3 / 5x5)。"""
    if rng.random() > p:
        return ink, "morph_keep"
    r = int(rng.integers(1, 3))                 # 1 -> 3x3, 2 -> 5x5
    st = ellipse_struct(r)
    op = "dilate" if rng.random() < 0.5 else "erode"
    binm = ink > 0.5
    out_b = binary_dilation(binm, structure=st) if op == "dilate" else binary_erosion(binm, structure=st)
    mode = "morph_thick" if op == "dilate" else "morph_thin"
    if not _area_ok(ink, out_b.astype(np.float32), mode):
        return ink, "morph_reject"
    # 保留原图的抗锯齿边缘: 只在二值层操作会变硬, 故用"距离场外侧软化"近似
    soft = gaussian_filter(out_b.astype(np.float32), 0.6)
    soft = np.clip(soft, 0.0, 1.0)
    return np.maximum(soft, ink * 0.0 + soft), f"morph_{op}{r}"


def _warp(ink, dx, dy):
    H, W = ink.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return map_coordinates(ink, [yy + dy, xx + dx], order=1, mode="constant", cval=0.0)


def aug_elastic(ink, rng, alpha=20.0, sigma=4.0):
    """② 弹性形变: 高斯平滑的随机位移场 (Simard et al.)。alpha 强度 / sigma 平滑度。"""
    H, W = ink.shape
    d = rng.uniform(-1.0, 1.0, size=(2, H, W)).astype(np.float32)
    d = np.stack([gaussian_filter(d[0], sigma) * alpha, gaussian_filter(d[1], sigma) * alpha])
    out = _warp(ink, d[0], d[1])
    if not _area_ok(ink, out, "geo"):
        return ink, "elastic_reject"
    return out, f"elastic_a{alpha:g}_s{sigma:g}"


def aug_grid(ink, rng, num_steps=4, distort_limit=0.1):
    """② 网格形变: (num_steps+1)^2 控制点抖动, 双线性插值成稠密位移场。"""
    H, W = ink.shape
    gx = np.linspace(0, W - 1, num_steps + 1)
    gy = np.linspace(0, H - 1, num_steps + 1)
    ox = rng.uniform(-distort_limit, distort_limit, size=(num_steps + 1, num_steps + 1)) * W
    oy = rng.uniform(-distort_limit, distort_limit, size=(num_steps + 1, num_steps + 1)) * H
    # 双线性插值到全分辨率
    from scipy.interpolate import RegularGridInterpolator
    xi = np.arange(W, dtype=np.float32)
    yi = np.arange(H, dtype=np.float32)
    fx = RegularGridInterpolator((gy, gx), ox, bounds_error=False, fill_value=0.0)
    fy = RegularGridInterpolator((gy, gx), oy, bounds_error=False, fill_value=0.0)
    XX, YY = np.meshgrid(xi, yi)
    pts = np.stack([YY.ravel(), XX.ravel()], axis=-1)
    dx = fx(pts).reshape(H, W).astype(np.float32)
    dy = fy(pts).reshape(H, W).astype(np.float32)
    out = _warp(ink, dx, dy)
    if not _area_ok(ink, out, "geo"):
        return ink, "grid_reject"
    return out, f"grid_n{num_steps}"


def aug_affine(ink, rng, shear_deg=3.0, scale=(0.95, 1.05), translate=0.02):
    """③ 限制性仿射: **禁止旋转**, 只做错切/微缩放/微平移。"""
    H, W = ink.shape
    s = float(rng.uniform(*scale))
    sh = math.radians(float(rng.uniform(-shear_deg, shear_deg)))
    tx = float(rng.uniform(-translate, translate)) * W
    ty = float(rng.uniform(-translate, translate)) * H
    # 以中心为原点: [x'] = s*R_shear*[x] + t
    A = np.array([[s, s * math.tan(sh)], [0.0, s]], dtype=np.float32)
    c = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    Ainv = np.linalg.inv(A)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx - c[0] - tx
    y = yy - c[1] - ty
    sx = Ainv[0, 0] * x + Ainv[0, 1] * y + c[0]
    sy = Ainv[1, 0] * x + Ainv[1, 1] * y + c[1]
    out = map_coordinates(ink, [sy, sx], order=1, mode="constant", cval=0.0)
    if not _area_ok(ink, out, "geo"):
        return ink, "affine_reject"
    return out, f"affine_s{shear_deg:g}_k{s:.3f}"


AUGS = {
    "morph": aug_morph,
    "elastic": aug_elastic,
    "grid": aug_grid,
    "affine": aug_affine,
}

# strategy -> 每张原图产出哪些变体（tag 前缀, 函数, kwargs）
STRATEGIES = {
    "geo3x": [("morph", aug_morph, {}), ("elastic", aug_elastic, {}), ("affine", aug_affine, {})],
    "geo5x": [("morph", aug_morph, {}), ("elastic", aug_elastic, {}), ("grid", aug_grid, {}),
              ("affine", aug_affine, {})],
    "elastic": [("elastic", aug_elastic, {})],
    "affine": [("affine", aug_affine, {})],
    "morphE": [("morph", aug_morph, {})],          # 椭圆核形态学（对齐文献 ①）
}
ID_BASE = {s: {t: 9_500_000 + i * 100_000 for i, (t, _, _) in enumerate(v)}
           for s, v in STRATEGIES.items()}


# ---------------------------------------------------------------- 自检拼图
def selftest(out_png, n=4, seed=0):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csvp = os.path.join(root, "exp-std/csv/eval200_fixed.csv")
    rows = [r for r in csv.DictReader(open(csvp, encoding="utf-8"))
            if os.path.exists(os.path.join(root, r["image_path"]))][:n]
    rng = np.random.default_rng(seed)
    fams = [("morph", aug_morph), ("elastic", aug_elastic), ("grid", aug_grid), ("affine", aug_affine)]
    tiles = []
    for r in rows:
        g = np.asarray(Image.open(os.path.join(root, r["image_path"])).convert("L"), dtype=np.uint8)
        row_tiles = [g]
        for _, fn in fams:
            out, tag = fn(_ink(g), rng)
            row_tiles.append(_to_gray(out))
        tiles.append(np.concatenate(row_tiles, axis=1))
    grid = np.concatenate(tiles, axis=0)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    Image.fromarray(grid).save(out_png)
    print(f"  selftest -> {out_png}  ({grid.shape[1]}x{grid.shape[0]})")
    print("  列顺序: 原图 | morph(椭圆核) | elastic | grid | affine")
    return 0


# ---------------------------------------------------------------- 批量生成
_G = {}


def _init(root, imgs_dir, jobs):
    _G.update(root=root, imgs_dir=imgs_dir, jobs=jobs)


def _one(task):
    idx, row = task
    root, imgs_dir, jobs = _G["root"], _G["imgs_dir"], _G["jobs"]
    # 每图独立 rng: 由 img_id 派生 -> 可复现
    iid = int(row["img_id"])
    rng = np.random.default_rng(iid * 2654435761 % (2 ** 32))
    src = os.path.join(root, row["image_path"])
    try:
        g = np.asarray(Image.open(src).convert("L"), dtype=np.uint8)
        if g.shape != (256, 256):
            g = np.asarray(Image.fromarray(g).resize((256, 256), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return idx, []
    out = []
    for tag, fn, kw in jobs:
        new_id = ID_BASE_STRAT[tag] + iid
        dst = os.path.join(root, imgs_dir, f"{new_id}.png")
        if not os.path.exists(dst):
            ink_new, how = fn(_ink(g), rng, **kw)
            Image.fromarray(_to_gray(ink_new)).save(dst)
        else:
            how = "cached"
        out.append((new_id, dst.replace(root + os.sep, "").replace("\\", "/"), tag, how))
    return idx, out


ID_BASE_STRAT = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-png", default="scratch/aug_geo_selftest.png")
    ap.add_argument("--strategy", default="geo3x", choices=sorted(STRATEGIES))
    ap.add_argument("--root", default=".")
    ap.add_argument("--src-csv")
    ap.add_argument("--out-csv")
    ap.add_argument("--imgs-dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 行（调试）")
    a = ap.parse_args()

    if a.selftest:
        return selftest(a.out_png)
    for f in ("src_csv", "out_csv", "imgs_dir"):
        if not getattr(a, f):
            ap.error(f"--{f.replace('_','-')} 必填（非 --selftest 模式）")

    global ID_BASE_STRAT
    ID_BASE_STRAT = {t: ID_BASE[a.strategy][t] for t, _, _ in STRATEGIES[a.strategy]}
    os.makedirs(os.path.join(a.root, a.imgs_dir), exist_ok=True)
    rows = list(csv.DictReader(open(a.src_csv, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    jobs = STRATEGIES[a.strategy]
    print(f"[aug-geo] strategy={a.strategy} tags={[t for t, _, _ in jobs]} src={len(rows):,} 行")

    import multiprocessing as mp
    t0 = time.time()
    res = []
    with mp.Pool(a.workers, initializer=_init, initargs=(a.root, a.imgs_dir, jobs)) as pool:
        for i, (idx, out) in enumerate(pool.imap_unordered(_one, list(enumerate(rows)), chunksize=64)):
            res.append((idx, out))
            if (i + 1) % 20000 == 0:
                print(f"  {i+1:,}/{len(rows):,}  ({time.time()-t0:.0f}s)")
    res.sort(key=lambda x: x[0])

    # 全景 CSV: 原图 + 各变体
    out_rows = []
    counts = {t: 0 for t, _, _ in jobs}
    for idx, variants in res:
        r0 = dict(rows[idx]); r0["aug"] = "orig"
        out_rows.append(r0)
        for nid, path, tag, how in variants:
            r = dict(rows[idx])
            r["img_id"] = str(nid)
            r["image_path"] = path
            r["aug"] = tag
            r["aug_from"] = str(rows[idx]["img_id"])
            out_rows.append(r)
            counts[tag] += 1
    ids = [int(r["img_id"]) for r in out_rows]
    assert len(set(ids)) == len(ids), "img_id 有重复!"
    with open(a.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    man = {"strategy": a.strategy, "src_csv": a.src_csv, "out_csv": a.out_csv,
           "imgs_dir": a.imgs_dir, "id_base": ID_BASE_STRAT, "counts": counts,
           "n_src": len(rows), "n_rows": len(out_rows),
           "geometry_preserving": a.strategy in ("morphE",),
           "repa_note": ("几何增强改变 DINO 特征 -> 增强行的 REPA 特征**必须重算**"
                         "（tools/ablation/expand_repa_cache.py）；形态学可复用"),
           "protection": {"thick_max_mult": THICK_MAX_MULT,
                          "thin_min_frac": THIN_MIN_FRAC, "thin_min_px": THIN_MIN_PX,
                          "geo_max_area_delta": GEO_MAX_AREA_DELTA},
           "created": datetime.now().isoformat(timespec="seconds")}
    json.dump(man, open(a.out_csv + ".manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[aug-geo] 写出 {a.out_csv} ({len(out_rows):,} 行) + manifest；图 -> {a.imgs_dir}")
    print(f"[aug-geo] 各变体计数: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
