#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_aug_geo.py — 书法专属数据增强构建器 v2（文献四方案的 ①②③）

v2 修复（2026-10-11，依据 aug_geo_audit 的定量结论）
----------------------------------------------------
* **拓扑闸门（最重要）**：连通分量数 或 封闭洞数 任一改变 → 拒收重采，最多 6 次，仍失败退回原图。
  这是"字没变"的真正保证（比面积保护更本质）。审计实测 morph 43.9% 改连通 / 40.6% 改洞数。
* **morph 不再二值化输出**：改为**软边界层** `ink ± 形态学差分`，保住原图抗锯齿。
* **grid 改平滑累积位移** + `distort_limit` 默认 0.03；旧版"每控制点独立抖 ±25.6px"实测 IoU 0.387。

三族（零新依赖: PIL + scipy）
----------------------------
  morph   : ① 形态学粗细抖动 —— 椭圆结构元（对齐 cv2.MORPH_ELLIPSE）
  elastic : ② 弹性形变 —— 高斯平滑随机位移场（Simard et al.）
  grid    : ② 网格形变 —— 4×4 控制点**累积**抖动
  affine  : ③ 限制性仿射 —— **禁止旋转**；±3° 错切 / 0.95~1.05 缩放 / ±2% 平移

几何变换都在**墨迹通道**（ink = 255 - gray）上重采样再反相 —— 插值在"无墨=0"一侧混合。

REPA 代价
---------
  形态学不改变 DINO patch-token → 增强行**可复用源图特征**；
  几何会改变 DINO 特征 → 应重算，或显式承担轻微错配。
  本脚本**沿用旧 ID 约定**（tp=7_000_000 / tn=7_100_000）→ 可直接复用 data/dino_cache/top10_aug_v1。
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
    binary_dilation, binary_erosion, gaussian_filter, label, map_coordinates,
)

THIN_MIN_FRAC, THIN_MIN_PX = 0.15, 20
THICK_MAX_MULT = 3.0
GEO_MAX_AREA_DELTA = 0.35
TOPO_TRIES = 6


def ellipse_struct(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2 + 1e-9


def _ink(gray):
    return 1.0 - gray.astype(np.float32) / 255.0


def _to_gray(ink):
    return np.clip((1.0 - ink) * 255.0, 0, 255).astype(np.uint8)


def _bin_stats(ink):
    b = ink > 0.5
    lab, n = label(b)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        nc = int((sizes >= 8).sum())
    else:
        nc = 0
    lab2, n2 = label(~b)
    border = set(lab2[0, :]) | set(lab2[-1, :]) | set(lab2[:, 0]) | set(lab2[:, -1])
    nh = int(sum(1 for i in range(1, n2 + 1) if i not in border))
    return int(b.sum()), nc, nh


def _area_ok(a0, a1, mode):
    if a0 < 1 or a1 < 1:
        return False
    if mode == "morph_thick":
        return a1 <= THICK_MAX_MULT * a0
    if mode == "morph_thin":
        return a1 >= THIN_MIN_FRAC * a0 and a1 >= THIN_MIN_PX
    return abs(a1 - a0) / max(a0, 1.0) <= GEO_MAX_AREA_DELTA


def _topo_ok(ink0, ink1):
    """★ 拓扑闸门: 连通分量数 / 封闭洞数 必须完全不变。"""
    a0, c0, h0 = _bin_stats(ink0)
    a1, c1, h1 = _bin_stats(ink1)
    return ((c0, h0) == (c1, h1)), (a0, a1)


def _gated(fn, tries=TOPO_TRIES):
    """包成"拓扑闸门 + 面积保护 + 重采"，对调用方完全透明。"""
    def wrapper(ink, rng, **kw):
        for _ in range(tries):
            out, tag = fn(ink, rng, **kw)
            ok, (a0, a1) = _topo_ok(ink, out)
            if not ok:
                continue
            mode = ("morph_thick" if "dilate" in tag else
                    "morph_thin" if "erode" in tag else "geo")
            if not _area_ok(a0, a1, mode):
                continue
            return out, tag
        return ink, "reject_topo"
    return wrapper


# ---------------------------------------------------------------- ① 形态学
def _morph_raw(ink, rng, p=1.0):
    if rng.random() > p:
        return ink, "morph_keep"
    r = int(rng.integers(1, 3))
    st = ellipse_struct(r)
    b = ink > 0.5
    if rng.random() < 0.5:
        newb = binary_dilation(b, structure=st).astype(np.float32)
        layer = np.clip(newb - b.astype(np.float32), 0.0, 1.0)
        return np.clip(ink + layer, 0.0, 1.0), f"morph_dilate{r}"
    newb = binary_erosion(b, structure=st).astype(np.float32)
    layer = np.clip(b.astype(np.float32) - newb, 0.0, 1.0)
    return np.clip(ink - layer, 0.0, 1.0), f"morph_erode{r}"


# ---------------------------------------------------------------- ② 几何
def _warp(ink, dx, dy):
    H, W = ink.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return map_coordinates(ink, [yy + dy, xx + dx], order=1, mode="constant", cval=0.0)


def _elastic_raw(ink, rng, alpha=12.0, sigma=4.0):
    H, W = ink.shape
    d = rng.uniform(-1.0, 1.0, size=(2, H, W)).astype(np.float32)
    return _warp(ink, gaussian_filter(d[0], sigma) * alpha,
                 gaussian_filter(d[1], sigma) * alpha), f"elastic_a{alpha:g}"


def _grid_raw(ink, rng, num_steps=4, distort_limit=0.03):
    """★ 累积随机游走: 沿网格逐点累积位移 → 形变平滑连续。"""
    from scipy.interpolate import RegularGridInterpolator
    H, W = ink.shape
    n = num_steps + 1
    step_x, step_y = W / num_steps, H / num_steps
    dxc = np.cumsum(rng.uniform(-distort_limit, distort_limit, size=(n, n)), axis=1) * step_x
    dyc = np.cumsum(rng.uniform(-distort_limit, distort_limit, size=(n, n)), axis=0) * step_y
    dxc -= dxc.mean()
    dyc -= dyc.mean()
    gx, gy = np.linspace(0, W - 1, n), np.linspace(0, H - 1, n)
    XX, YY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([YY.ravel(), XX.ravel()], axis=-1)
    fx = RegularGridInterpolator((gy, gx), dxc, bounds_error=False, fill_value=0.0)
    fy = RegularGridInterpolator((gy, gx), dyc, bounds_error=False, fill_value=0.0)
    return _warp(ink, fx(pts).reshape(H, W).astype(np.float32),
                 fy(pts).reshape(H, W).astype(np.float32)), f"grid_n{num_steps}"


def _affine_raw(ink, rng, shear_deg=3.0, scale=(0.95, 1.05), translate=0.02):
    H, W = ink.shape
    s = float(rng.uniform(*scale))
    sh = math.radians(float(rng.uniform(-shear_deg, shear_deg)))
    tx = float(rng.uniform(-translate, translate)) * W
    ty = float(rng.uniform(-translate, translate)) * H
    A = np.array([[s, s * math.tan(sh)], [0.0, s]], dtype=np.float32)
    c = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    Ainv = np.linalg.inv(A)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = xx - c[0] - tx, yy - c[1] - ty
    sx = Ainv[0, 0] * x + Ainv[0, 1] * y + c[0]
    sy = Ainv[1, 0] * x + Ainv[1, 1] * y + c[1]
    return map_coordinates(ink, [sy, sx], order=1, mode="constant", cval=0.0), f"affine_s{shear_deg:g}"


aug_morph = _gated(_morph_raw)
aug_elastic = _gated(_elastic_raw)
aug_grid = _gated(_grid_raw)
aug_affine = _gated(_affine_raw)

AUGS = {"morph": aug_morph, "elastic": aug_elastic, "grid": aug_grid, "affine": aug_affine}

# 策略 -> [(tag, fn, kwargs)]；沿用旧 ID 约定 tp/tn 以复用 dino_cache/top10_aug_v1
STRATEGIES = {
    "morphE":  [("tp", aug_morph, {}), ("tn", aug_morph, {})],
    "elastic": [("tp", aug_elastic, {}), ("tn", aug_elastic, {})],
    "grid":    [("tp", aug_grid, {}), ("tn", aug_grid, {})],
    "affine":  [("tp", aug_affine, {}), ("tn", aug_affine, {})],
}
ID_BASE = {s: {t: (7_000_000 if t == "tp" else 7_100_000) for t, _, _ in v}
           for s, v in STRATEGIES.items()}


def selftest(out_png, n=4, seed=0):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = [r for r in csv.DictReader(open(os.path.join(root, "exp-std/csv/eval200_fixed.csv"),
                                          encoding="utf-8"))
            if os.path.exists(os.path.join(root, r["image_path"]))][:n]
    rng = np.random.default_rng(seed)
    fams = [("morph", aug_morph), ("elastic", aug_elastic), ("grid", aug_grid), ("affine", aug_affine)]
    tiles = []
    for r in rows:
        g = np.asarray(Image.open(os.path.join(root, r["image_path"])).convert("L"), dtype=np.uint8)
        row_tiles = [g]
        for _, fn in fams:
            out, _tag = fn(_ink(g), rng)
            row_tiles.append(_to_gray(out))
        tiles.append(np.concatenate(row_tiles, axis=1))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=0)).save(out_png)
    print(f"  selftest -> {out_png}  列: 原图 | morph | elastic | grid | affine")
    return 0


_G = {}


def _init(root, imgs_dir, jobs):
    _G.update(root=root, imgs_dir=imgs_dir, jobs=jobs)


def _one(task):
    idx, row = task
    root, imgs_dir, jobs = _G["root"], _G["imgs_dir"], _G["jobs"]
    iid = int(row["img_id"])
    rng = np.random.default_rng((iid * 2654435761) % (2 ** 32))
    try:
        g = np.asarray(Image.open(os.path.join(root, row["image_path"])).convert("L"), dtype=np.uint8)
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
    ap.add_argument("--strategy", default="morphE", choices=sorted(STRATEGIES))
    ap.add_argument("--root", default=".")
    ap.add_argument("--src-csv")
    ap.add_argument("--out-csv")
    ap.add_argument("--imgs-dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.out_png)
    for f in ("src_csv", "out_csv", "imgs_dir"):
        if not getattr(a, f):
            ap.error(f"--{f.replace('_','-')} 必填")

    global ID_BASE_STRAT
    ID_BASE_STRAT = dict(ID_BASE[a.strategy])
    os.makedirs(os.path.join(a.root, a.imgs_dir), exist_ok=True)
    rows = list(csv.DictReader(open(a.src_csv, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    jobs = STRATEGIES[a.strategy]
    print(f"[aug-geo] strategy={a.strategy} tags={[t for t, _, _ in jobs]} src={len(rows):,} 行",
          flush=True)

    import multiprocessing as mp
    t0 = time.time()
    res = []
    with mp.Pool(a.workers, initializer=_init, initargs=(a.root, a.imgs_dir, jobs)) as pool:
        for i, (idx, out) in enumerate(pool.imap_unordered(_one, list(enumerate(rows)), chunksize=64)):
            res.append((idx, out))
            if (i + 1) % 20000 == 0:
                print(f"  {i+1:,}/{len(rows):,} ({time.time()-t0:.0f}s)", flush=True)
    res.sort(key=lambda x: x[0])

    out_rows, counts, rejects = [], {t: 0 for t, _, _ in jobs}, 0
    for idx, variants in res:
        r0 = dict(rows[idx]); r0["aug"] = "orig"; r0["aug_from"] = ""
        out_rows.append(r0)
        for nid, path, tag, how in variants:
            r = dict(rows[idx])
            r["img_id"] = str(nid); r["image_path"] = path; r["aug"] = tag
            r["aug_from"] = str(rows[idx]["img_id"])
            out_rows.append(r); counts[tag] += 1
            if how == "reject_topo":
                rejects += 1
    ids = [int(r["img_id"]) for r in out_rows]
    assert len(set(ids)) == len(ids), "img_id 有重复!"
    with open(a.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    man = {"strategy": a.strategy, "src_csv": a.src_csv, "out_csv": a.out_csv,
           "imgs_dir": a.imgs_dir, "id_base": ID_BASE_STRAT, "counts": counts,
           "n_src": len(rows), "n_rows": len(out_rows), "n_reject_topo": rejects,
           "reject_rate": round(rejects / max(len(rows) * len(jobs), 1), 4),
           "topo_gate": "连通分量数/封闭洞数 任一改变即拒收重采(最多6次)",
           "protection": {"thick_max_mult": THICK_MAX_MULT, "thin_min_frac": THIN_MIN_FRAC,
                          "geo_max_area_delta": GEO_MAX_AREA_DELTA},
           "repa_note": "形态学可复用源图 DINO 特征; 几何沿用 tp/tn 旧 ID 以复用 top10_aug_v1",
           "created": datetime.now().isoformat(timespec="seconds")}
    json.dump(man, open(a.out_csv + ".manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[aug-geo] 写出 {a.out_csv} ({len(out_rows):,} 行)；拓扑拒收率 {man['reject_rate']:.2%}")
    print(f"[aug-geo] 计数 {counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
