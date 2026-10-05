#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_top10_sym_aug.py — 为 top10 数据集生成 v4 对称笔画粗细增强图像与全量 CSV

原理 (v4 对称笔画增强):
  针对每张原图，生成一对严格对偶的 ±同幅 变体:
    tp (thicken / 变粗) = binary_dilation(ink, ST, iterations=p)
    tn (thin / 变细)    = binary_erosion(ink, ST, iterations=p)
  同一原图的 ± 对采用相同的 p ∈ {1, 2} (确定性哈希)。
  保护逻辑:
    - 极端细断裂保护: 若 tn 面积 < 原墨迹面积的 15% 或 < 20px，降档 p=1；若仍不足则跳过 tn
    - 极端粗糊块保护: 若 tp 面积 > 原墨迹面积的 3.0 倍，降档 p=1；若仍过大则跳过 tp
  规整尺寸:
    - 严格保持 256x256 白纸黑墨 (背景 255, 墨迹 0, L 模式 PNG)

输入: /root/Workspace/xy/DiT/exp-std/csv/train.csv (26,002 条)
输出:
  图像目录: /root/Workspace/xy/DiT/data/top10_style23/imgs_aug_sym/
  清单文件: /root/Workspace/xy/DiT/exp-std/csv/train_top10_aug_sym.csv (~78,000 条)
"""

import csv
import multiprocessing as mp
import os
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = os.path.join(ROOT, "exp-std/csv/train.csv")
OUT_DIR = os.path.join(ROOT, "data/top10_style23/imgs_aug_sym")
OUT_CSV = os.path.join(ROOT, "exp-std/csv/train_top10_aug_sym.csv")

os.makedirs(OUT_DIR, exist_ok=True)
ST = generate_binary_structure(2, 2)  # 8-邻域结构元


def process_row(task):
    """task: (idx, row) -> (idx, {'tp': rel_path|None, 'tn': rel_path|None})"""
    idx, row = task
    orig_rel_path = row["image_path"]
    orig_abs_path = os.path.join(ROOT, orig_rel_path)

    # 目标路径
    img_id = str(row.get("img_id", f"{idx:06d}")).zfill(6)
    tp_fname = f"{img_id}_tp.png"
    tn_fname = f"{img_id}_tn.png"
    tp_abs = os.path.join(OUT_DIR, tp_fname)
    tn_abs = os.path.join(OUT_DIR, tn_fname)
    tp_rel = os.path.relpath(tp_abs, ROOT)
    tn_rel = os.path.relpath(tn_abs, ROOT)

    out = {"tp": None, "tn": None}

    # 幂等跳过
    if os.path.exists(tp_abs) and os.path.exists(tn_abs):
        return idx, {"tp": tp_rel, "tn": tn_rel}

    try:
        im = Image.open(orig_abs_path).convert("L")
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.BICUBIC)
        g = np.asarray(im)
    except Exception as e:
        return idx, out

    ink = g < 128
    orig_area = int(ink.sum())
    if orig_area < 20:
        return idx, out

    # 确定性 p ∈ {1, 2}
    p = 1 + ((idx * 2654435761) % 2)

    # 1. Thicken (tp)
    pp_t = p
    ok_t = False
    m_tp = None
    while pp_t > 0:
        m_tp = binary_dilation(ink, structure=ST, iterations=pp_t)
        if int(m_tp.sum()) <= 3.0 * orig_area:
            ok_t = True
            break
        pp_t -= 1

    if ok_t and m_tp is not None:
        tp_arr = np.where(m_tp, 0, 255).astype(np.uint8)
        Image.fromarray(tp_arr, mode="L").save(tp_abs, "PNG", compress_level=1)
        out["tp"] = tp_rel

    # 2. Thin (tn)
    pp_n = p
    ok_n = False
    m_tn = None
    while pp_n > 0:
        m_tn = binary_erosion(ink, structure=ST, iterations=pp_n)
        a_n = int(m_tn.sum())
        if a_n >= 0.15 * orig_area and a_n >= 20:
            ok_n = True
            break
        pp_n -= 1

    if ok_n and m_tn is not None:
        tn_arr = np.where(m_tn, 0, 255).astype(np.uint8)
        Image.fromarray(tn_arr, mode="L").save(tn_abs, "PNG", compress_level=1)
        out["tn"] = tn_rel

    return idx, out


def main():
    print("=" * 80)
    print("【开始生成 top10 对称笔画粗细增强图像 (v4 Symmetric Augmentation)】")
    print(f"  源 CSV: {SRC_CSV}")
    print(f"  增强图目标目录: {OUT_DIR}")
    print(f"  增强 CSV 目标: {OUT_CSV}")
    print("=" * 80)

    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
    n_src = len(rows)
    print(f"原始训练集行数: {n_src:,} 条")

    tasks = list(enumerate(rows))
    t0 = time.time()
    results = {}

    workers = min(48, mp.cpu_count())
    print(f"启动 {workers} 个并发 Worker 进行双向膨胀/腐蚀计算与 PNG 落盘...")

    with mp.Pool(workers) as pool:
        for idx, out in pool.imap_unordered(process_row, tasks, chunksize=128):
            results[idx] = out
            if (idx + 1) % 5000 == 0 or (idx + 1) == n_src:
                elapsed = time.time() - t0
                print(
                    f"  进度: {idx + 1:,}/{n_src:,} ({float(idx + 1)/elapsed:.1f} 图/秒)"
                )

    dt = time.time() - t0
    n_tp = sum(1 for v in results.values() if v.get("tp"))
    n_tn = sum(1 for v in results.values() if v.get("tn"))

    print("=" * 80)
    print(f"增强图生成完毕！总耗时: {dt:.1f} 秒")
    print(f"  原图数量 : {n_src:,} 张")
    print(f"  变粗(tp) : {n_tp:,} 张 (成功率 {n_tp/n_src*100:.2f}%)")
    print(f"  变细(tn) : {n_tn:,} 张 (成功率 {n_tn/n_src*100:.2f}%)")
    print(f"  总物理图像: {n_src + n_tp + n_tn:,} 张 (扩增约 3.0 倍)")
    print("=" * 80)

    # 构建并输出增强全景 train_top10_aug_sym.csv
    fields = list(rows[0].keys())
    if "aug" not in fields:
        fields.append("aug")

    out_rows = []
    for idx, r in enumerate(rows):
        # 1. 原图
        r_orig = dict(r)
        r_orig["aug"] = ""
        out_rows.append(r_orig)

        # 2. 增强变体
        aug_res = results.get(idx, {})
        for aug_tag in ("tp", "tn"):
            aug_p = aug_res.get(aug_tag)
            if aug_p:
                r_aug = dict(r)
                r_aug["image_path"] = aug_p
                r_aug["aug"] = aug_tag
                r_aug["img_id"] = (
                    f"{r['img_id']}_{aug_tag}"
                    if "img_id" in r
                    else f"{idx:06d}_{aug_tag}"
                )
                out_rows.append(r_aug)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(
        f"增强 CSV 成功写入: {OUT_CSV} (共 {len(out_rows):,} 行, 包含原图 + tp + tn)"
    )
    print("★ VAE Latent Encoding 已按指示暂缓，等当前训练阶段结束后再行编码！")
    print("=" * 80)


if __name__ == "__main__":
    main()
