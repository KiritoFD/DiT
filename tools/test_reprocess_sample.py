# -*- coding: utf-8 -*-
"""test_reprocess_sample.py — 测试从原始拓片进行古法CV自适应清洗+标准化的效果。"""
import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def create_line_kernel(angle_deg, length):
    rad = np.deg2rad(angle_deg)
    dx = np.cos(rad)
    dy = np.sin(rad)
    half = length // 2
    pts = []
    for step in range(-half, half + 1):
        x = int(round(step * dx))
        y = int(round(step * dy))
        pts.append((x, y))
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    k = np.zeros((h, w), dtype=np.uint8)
    for x, y in pts:
        k[y - min_y, x - min_x] = 1
    return k

LINE_KERNELS = [create_line_kernel(deg, 7) for deg in [0, 90, 45, 135]]

def process_raw_calligraphy(raw_path, target_size=256, box_frac=0.88):
    """从原始拓片图/原图到纯净标准 256x256 二值书法图。"""
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None, "read_fail"

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

    # 若墨色极淡导致 Otsu 异常，做兜底检查
    ink_ratio = float(ink.mean()) / 255.0
    if ink_ratio < 0.005 or ink_ratio > 0.85:
        # 降级到 Sauvola 或局部自适应
        local_th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 10)
        ink = (local_th == 0).astype(np.uint8) * 255

    # 3. 几何杂质过滤 (去除石碑边界线与孤立噪点)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    main_mask = np.zeros_like(ink, dtype=bool)
    for i in range(1, num_labels):
        x, y, sw, sh, area = stats[i]
        # 跳过边缘细线
        if (x <= 5 or x + sw >= w - 5) and (sh > 80 and sw <= 5):
            continue
        if (y <= 5 or y + sh >= h - 5) and (sw > 80 and sh <= 5):
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
        # 边界线
        if (x <= 5 or x + sw >= w - 5) and (sh > 80 and sw <= 5):
            continue
        if (y <= 5 or y + sh >= h - 5) and (sw > 80 and sh <= 5):
            continue
        # 孤立浮尘
        comp_dist = dist_to_main[labels == i].min() if main_mask.any() else 0
        if area < 20 and comp_dist > 8.0:
            continue
        if area < 8:
            continue
        clean_ink[labels == i] = 255

    # 4. 方向性微愈合 (多角度 1px 细线闭运算，顺应笔势补齐微隙，不增粗笔画)
    for k in LINE_KERNELS:
        closed = cv2.morphologyEx(clean_ink, cv2.MORPH_CLOSE, k)
        clean_ink = cv2.bitwise_or(clean_ink, closed)

    # 5. 标准化几何布局 (256x256, 居中, box_frac=0.88, 白底黑字)
    m = clean_ink > 0
    if not m.any():
        return None, "empty_ink"

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

    return canvas, "ok"

if __name__ == "__main__":
    # Test on raw Hui
    res, status = process_raw_calligraphy(
        "C:/Users/xy/.gemini/antigravity/brain/10d772c0-e343-4844-99e1-580c6676138e/scratch/hui_raw_wild.png"
    )
    if res is not None:
        cv2.imwrite("C:/Users/xy/.gemini/antigravity/brain/10d772c0-e343-4844-99e1-580c6676138e/scratch/test_pipeline_hui.png", res)
        print("Success: saved test_pipeline_hui.png, status:", status)
