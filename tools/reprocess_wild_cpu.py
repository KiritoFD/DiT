# -*- coding: utf-8 -*-
"""reprocess_wild_cpu.py — CPU 并行管线：从原始拓片重新处理 HCSU wild 样本，修复断笔与碎裂问题。

设计目标：
  1. 彻底解决粗暴硬阈值 (a < 128) 导致的断笔、空洞、碎裂与椒盐噪点。
  2. 采用「保边滤波 + 动态双峰 Otsu / Sauvola 自适应二值化 + 连通域几何去噪 + 256x256 标准几何归一化」。
  3. 纯 CPU 多进程执行（默认 32 workers），零 GPU 占用，不干扰后台正在进行的 V22 训练。
  4. 产物落盘至独立目录并输出完整追踪 CSV，供后续 encode 替换 shard 使用。

用法:
  python tools/reprocess_wild_cpu.py [--workers 32] [--limit 100]
  python tools/reprocess_wild_cpu.py --workers 32
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

TARGET_SIZE = 256
BOX_FRAC = 0.88


def clean_and_normalize(raw_path, target_size=TARGET_SIZE, box_frac=BOX_FRAC):
    """从原始拓片图/原图到纯净标准 256x256 二值书法图。"""
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None, "read_fail", 0.0

    h, w = raw.shape
    g = raw.astype(np.float32) / 255.0
    border = float(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]).mean())

    # 极性判定: 统计相比边框底色更暗与更亮的像素数量 (墨迹/笔画是少数派)
    n_dark = int((g < border - 0.20).sum())
    n_light = int((g > border + 0.20).sum())
    if n_dark + n_light < 0.001 * g.size:
        inverted = border < 0.5
    else:
        inverted = n_light > n_dark

    inv = 255 - raw if inverted else raw

    # 1. 保边去噪 (去除石纹、宣纸纤维、浮尘，保留笔锋边缘)
    denoised = cv2.bilateralFilter(inv, d=5, sigmaColor=35, sigmaSpace=35)

    # 2. Otsu 自适应二值化 (根据笔墨与纸面底色双峰分布动态自适应，不割断笔腰)
    _, otsu_bin = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (otsu_bin == 0).astype(np.uint8) * 255

    # 若墨色极淡导致 Otsu 异常，做局部自适应兜底
    ink_ratio_raw = float(ink.mean()) / 255.0
    if ink_ratio_raw < 0.005 or ink_ratio_raw > 0.85:
        local_th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 10)
        ink = (local_th == 0).astype(np.uint8) * 255

    # 3. 几何杂质过滤 (去除石碑边界线与孤立噪点)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    main_mask = np.zeros_like(ink, dtype=bool)
    for i in range(1, num_labels):
        x, y, sw, sh, area = stats[i]
        # 跳过边缘细线 (长宽比极端且贴近边缘)
        if (x <= 6 or x + sw >= w - 6) and (sh > 60 and sw <= 6):
            continue
        if (y <= 6 or y + sh >= h - 6) and (sw > 60 and sh <= 6):
            continue
        if area >= 50:
            main_mask[labels == i] = True

    if main_mask.any():
        dist_to_main = distance_transform_edt(~main_mask)
    else:
        dist_to_main = np.zeros_like(ink, dtype=np.float32)

    clean_ink = np.zeros_like(ink)
    for i in range(1, num_labels):
        x, y, sw, sh, area = stats[i]
        # 边界线过滤
        if (x <= 6 or x + sw >= w - 6) and (sh > 60 and sw <= 6):
            continue
        if (y <= 6 or y + sh >= h - 6) and (sw > 60 and sh <= 6):
            continue
        # 孤立浮尘颗粒过滤 (面积小且远离主字形)
        comp_dist = dist_to_main[labels == i].min() if main_mask.any() else 0
        if area < 15:
            continue
        if area < 30 and comp_dist > 8.0:
            continue
        clean_ink[labels == i] = 255

    # 4. 标准化几何布局 (256x256, 居中, box_frac=0.88, 白底黑字)
    m = clean_ink > 0
    if not m.any():
        return None, "empty_ink", 0.0

    ys, xs = np.where(m)
    crop = clean_ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    target_span = box_frac * target_size
    scale = target_span / max(crop.shape)
    tw = max(1, int(round(crop.shape[1] * scale)))
    th = max(1, int(round(crop.shape[0] * scale)))

    resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
    off_x = (target_size - tw) // 2
    off_y = (target_size - th) // 2

    ink_mask = resized > 100
    canvas[off_y:off_y + th, off_x:off_x + tw][ink_mask] = 0

    final_ink_ratio = float(ink_mask.mean())
    return canvas, "ok", final_ink_ratio


def _worker_task(item):
    idx, row, source_tag, raw_root, out_root = item
    src_p = str(row.get("src_image_path", ""))
    tag = source_tag.replace("hcsu_", "")
    prefix = f"data/hcsu_kxl/imgs/{tag}/"
    rel = src_p.replace(prefix, "")
    raw_p = os.path.join(raw_root, rel)
    dst_p = os.path.join(out_root, tag, rel)

    if not os.path.exists(raw_p):
        return {
            "idx": idx,
            "old_50k_id": row.get("old_50k_id"),
            "character": row.get("character"),
            "calligrapher": row.get("calligrapher"),
            "script": row.get("script"),
            "old_image_path": row.get("image_path"),
            "src_image_path": src_p,
            "raw_image_path": raw_p,
            "new_image_path": "",
            "ink_ratio": 0.0,
            "status": "raw_missing",
        }

    img, status, ink_ratio = clean_and_normalize(raw_p)
    if img is not None:
        os.makedirs(os.path.dirname(dst_p), exist_ok=True)
        cv2.imwrite(dst_p, img)
        return {
            "idx": idx,
            "old_50k_id": row.get("old_50k_id"),
            "character": row.get("character"),
            "calligrapher": row.get("calligrapher"),
            "script": row.get("script"),
            "old_image_path": row.get("image_path"),
            "src_image_path": src_p,
            "raw_image_path": raw_p,
            "new_image_path": dst_p,
            "ink_ratio": round(ink_ratio, 5),
            "status": "ok",
        }
    else:
        return {
            "idx": idx,
            "old_50k_id": row.get("old_50k_id"),
            "character": row.get("character"),
            "calligrapher": row.get("calligrapher"),
            "script": row.get("script"),
            "old_image_path": row.get("image_path"),
            "src_image_path": src_p,
            "raw_image_path": raw_p,
            "new_image_path": "",
            "ink_ratio": 0.0,
            "status": status,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_50k_v2_fixed_augmented.csv")
    ap.add_argument("--source-tag", default="hcsu_wild")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--out-root", default="/root/Workspace/xy/DiT/data/hcsu_reprocessed_clean")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = args.source_tag.replace("hcsu_", "")
    if not args.raw_root:
        args.raw_root = f"/root/Workspace/xy/HCSU/{tag}_extract"
    if not args.out_csv:
        args.out_csv = f"assets/reprocessed_{args.source_tag}.csv"

    print(f"=== Reprocess {args.source_tag} CPU Pipeline ===")
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    sub_df = df[df["source"] == args.source_tag].copy()
    if args.limit > 0:
        sub_df = sub_df.head(args.limit)
    total = len(sub_df)
    print(f"Total samples to process: {total}")

    tasks = []
    for idx, (_, row) in enumerate(sub_df.iterrows()):
        tasks.append((idx, row.to_dict(), args.source_tag, args.raw_root, args.out_root))

    t0 = time.time()
    results = []
    print(f"Starting multiprocessing pool with {args.workers} workers...")
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize=50)):
            results.append(res)
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                print(f"[{i + 1}/{total}] ({speed:.1f} samples/s) - Elapsed: {elapsed:.1f}s")

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("idx").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\nProcessing complete in {time.time() - t0:.1f}s!")
    print(f"Results saved to: {args.out_csv}")
    print("Status summary:")
    print(out_df["status"].value_counts())


if __name__ == "__main__":
    main()
