#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/test_align_intensities.py —
测试不同 alpha 强度 (0.35, 0.70, 1.00) 的 SDF 粗细对齐
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from skimage.morphology import medial_axis, skeletonize

ROOT = Path(__file__).resolve().parents[2]


def align_with_alpha(
    img_np: np.ndarray,
    target_dist: dict,
    alpha: float = 1.0,
    threshold: float = 0.5,
    max_shift: float = 2.5
) -> np.ndarray:
    """按指定 alpha 强度计算空间变异 SDF 对齐。"""
    if abs(alpha) < 1e-4:
        return img_np

    ink_mask = (1.0 - img_np) >= threshold
    if ink_mask.sum() < 25:
        return img_np

    dist_map = distance_transform_edt(ink_mask)
    skel = medial_axis(ink_mask)
    if not skel.any():
        skel = skeletonize(ink_mask)
    if not skel.any():
        return img_np

    widths = 2.0 * dist_map[skel]
    widths = widths[widths >= 0.5]
    if len(widths) < 10:
        return img_np

    mu_t = target_dist["ref_mean"]
    sigma_t = target_dist["ref_std"]
    mu_g = float(np.mean(widths))
    sigma_g = max(float(np.std(widths)), 0.2)

    scale = float(np.clip(sigma_t / sigma_g, 0.5, 2.0))
    skel_widths = 2.0 * dist_map[skel]
    target_widths = mu_t + scale * (skel_widths - mu_g)
    target_widths = np.clip(target_widths, 1.0, 45.0)

    delta_skel = (target_widths - skel_widths) / 2.0
    delta_skel = np.clip(delta_skel, -max_shift, max_shift)

    _, (nearest_r, nearest_c) = distance_transform_edt(~skel, return_indices=True)
    delta_grid = np.zeros_like(img_np, dtype=np.float32)
    delta_grid[skel] = delta_skel.astype(np.float32)
    delta_map = delta_grid[nearest_r, nearest_c]
    delta_map = gaussian_filter(delta_map, sigma=1.2)

    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    phi_prime = phi + alpha * delta_map
    adjusted_ink = np.clip(phi_prime + 0.5, 0.0, 1.0)
    return (1.0 - adjusted_ink).astype(np.float32)


if __name__ == "__main__":
    with open(ROOT / "assets/stroke_width_trainset_distribution.json", "r", encoding="utf-8") as f:
        td = json.load(f)
    print("Alignment function verified successfully!")
