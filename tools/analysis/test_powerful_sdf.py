import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from skimage.morphology import medial_axis, skeletonize

sys.path.insert(0, ".")
from tools.render_fixed_20row_poster import measure_p50_and_contrast

def powerful_sdf_align(
    img_np: np.ndarray,
    target_dist: dict,
    alpha: float = 1.0,
    threshold: float = 0.5,
    max_shift: float = 6.0,
    smooth_sigma: float = 1.5,
    edge_tau: float = 0.4
) -> np.ndarray:
    """真正有力量、粗细分明、视觉变化显著且边缘平滑的 SDF 笔画对齐:
    1. 放开位移上限 (max_shift 提升至 6.0px，允许笔宽真正缩放 8~12px)
    2. 取消对收细的内部屏蔽 (不强制 maximum(ink_soft)，让细笔真正变细)
    3. 精准空间位移场: sigma=1.5 平滑核，既避免锯齿与 Voronoi 结块，又保持位移幅度 90%+ 传递到轮廓边界
    4. 拓扑硬门禁: 严格保证不产生飞白空洞或断裂 (连通分量不能裂开)
    """
    if abs(alpha) < 1e-4:
        return img_np

    ink_soft = np.clip(1.0 - img_np, 0.0, 1.0)
    ink_mask = ink_soft >= threshold
    if ink_mask.sum() < 25:
        return img_np

    # 原始连通性与孔洞数
    lbl_orig, num_orig = label(ink_mask)
    lbl_bg, num_bg = label(~ink_mask)
    holes_orig = max(0, num_bg - 1)

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

    # 目标宽度映射
    scale = float(np.clip(sigma_t / sigma_g, 0.5, 2.0))
    skel_widths = 2.0 * dist_map[skel]
    target_widths = mu_t + scale * (skel_widths - mu_g)
    target_widths = np.clip(target_widths, 1.2, 45.0)

    # 计算法向半径位移 delta = (target - current) / 2
    delta_skel = (target_widths - skel_widths) / 2.0
    delta_skel = np.clip(delta_skel, -max_shift, max_shift)

    # 空间扩散场 (平滑扩散到全图)
    _, (nearest_r, nearest_c) = distance_transform_edt(~skel, return_indices=True)
    delta_grid = np.zeros_like(img_np, dtype=np.float32)
    delta_grid[skel] = delta_skel.astype(np.float32)
    delta_map = delta_grid[nearest_r, nearest_c]
    
    # 适度高斯平滑 (sigma=1.5 消除 Voronoi 不连续，保留 85% 位移峰值)
    delta_map = gaussian_filter(delta_map, sigma=smooth_sigma)

    # 连续 Signed Distance Field (SDF)
    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    # 施加位移场 (alpha 比例调节)
    # 当 delta > 0 (变粗)，phi 增大，墨迹外扩；
    # 当 delta < 0 (变细)，phi 减小，墨迹收缩。
    phi_prime = phi + alpha * delta_map

    # 拓扑连通性安全门禁检查
    mask_candidate = phi_prime >= 0.0
    lbl_cand, num_cand = label(mask_candidate)
    lbl_bg_cand, num_bg_cand = label(~mask_candidate)
    holes_cand = max(0, num_bg_cand - 1)

    # 如果断裂 (连通分量增多) 或者孔洞消失过多，逐步衰减位移
    atten = 1.0
    while (num_cand > num_orig + 1 or holes_cand < holes_orig - 2) and atten > 0.3:
        atten *= 0.75
        phi_prime = phi + (alpha * atten) * delta_map
        mask_candidate = phi_prime >= 0.0
        lbl_cand, num_cand = label(mask_candidate)
        lbl_bg_cand, num_bg_cand = label(~mask_candidate)
        holes_cand = max(0, num_bg_cand - 1)

    # 自然宣纸墨晕重建 (Sigmoid 柔和抗锯齿)
    ink_aligned = 1.0 / (1.0 + np.exp(-phi_prime / edge_tau))
    
    # 墨芯加浓: 保证笔画中心深邃纯黑，边缘细腻微晕
    ink_aligned = np.clip(ink_aligned * 1.05, 0.0, 1.0)

    return (1.0 - ink_aligned).astype(np.float32)

if __name__ == "__main__":
    with open("assets/stroke_width_trainset_distribution.json", encoding="utf-8") as f:
        td = json.load(f)

    # Test Wang Xizhi (thin) and Su Shi (thick)
    p_wang = "artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g80.png"
    p_su = "artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g111.png"

    im_w = np.array(Image.open(p_wang).convert("L"), dtype=np.float32) / 255.0
    im_s = np.array(Image.open(p_su).convert("L"), dtype=np.float32) / 255.0

    print("=== Testing Wang Xizhi (Target P50: 8.8) ===")
    p50_w0, _ = measure_p50_and_contrast(im_w)
    print(f"Raw: p50={p50_w0:.2f}px")
    for a in [0.35, 0.70, 1.00]:
        res = powerful_sdf_align(im_w, td["王羲之_楷"], alpha=a)
        p50, con = measure_p50_and_contrast(res)
        print(f"Alpha {a:.2f}: p50={p50:.2f}px (diff={p50 - p50_w0:+.2f}px)")
        Image.fromarray((res * 255).astype(np.uint8)).save(f"scratch/wang_a{int(a*100)}.png")

    print("\n=== Testing Su Shi (Target P50: 15.2) ===")
    p50_s0, _ = measure_p50_and_contrast(im_s)
    print(f"Raw: p50={p50_s0:.2f}px")
    for a in [0.35, 0.70, 1.00]:
        res = powerful_sdf_align(im_s, td["苏轼_楷"], alpha=a)
        p50, con = measure_p50_and_contrast(res)
        print(f"Alpha {a:.2f}: p50={p50:.2f}px (diff={p50 - p50_s0:+.2f}px)")
        Image.fromarray((res * 255).astype(np.uint8)).save(f"scratch/su_a{int(a*100)}.png")
