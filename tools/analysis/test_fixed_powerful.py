from PIL import Image
import numpy as np
import json
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from skimage.morphology import medial_axis, skeletonize

def count_structural_holes(mask: np.ndarray, min_area: int = 15) -> int:
    lbl_bg, num_bg = label(~mask)
    if num_bg <= 1:
        return 0
    sizes = np.bincount(lbl_bg.ravel())
    outer_bg = np.argmax(sizes)
    real_holes = sum(sizes[k] >= min_area for k in range(1, num_bg + 1) if k != outer_bg)
    return real_holes

def test_fixed_powerful_align(
    img_np: np.ndarray,
    target_dist: dict,
    alpha: float = 1.0,
    threshold: float = 0.5,
    max_shift: float = 7.0,
    smooth_sigma: float = 1.2,
    edge_tau: float = 0.35
) -> np.ndarray:
    if abs(alpha) < 1e-4:
        return img_np

    ink_soft = np.clip(1.0 - img_np, 0.0, 1.0)
    ink_mask = ink_soft >= threshold
    if ink_mask.sum() < 25:
        return img_np

    # 结构孔洞 (>= 15px) 与有效连通分量 (>= 20px)
    lbl_orig, num_orig = label(ink_mask)
    orig_sizes = np.bincount(lbl_orig.ravel())[1:]
    major_orig_comp = sum(s >= 20 for s in orig_sizes)
    real_holes_orig = count_structural_holes(ink_mask, min_area=15)

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
    delta_map = gaussian_filter(delta_map, sigma=smooth_sigma)

    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    phi_prime = phi + alpha * delta_map

    # 门禁检查: 绝不能断裂大笔画 (主要连通分量增多)
    mask_c = phi_prime >= 0.0
    lbl_c, num_c = label(mask_c)
    cand_sizes = np.bincount(lbl_c.ravel())[1:] if num_c > 0 else []
    major_cand_comp = sum(s >= 20 for s in cand_sizes)
    real_holes_c = count_structural_holes(mask_c, min_area=15)

    atten = 1.0
    while (major_cand_comp > major_orig_comp + 1 or real_holes_c < real_holes_orig - 1) and atten > 0.3:
        atten *= 0.8
        phi_prime = phi + (alpha * atten) * delta_map
        mask_c = phi_prime >= 0.0
        lbl_c, num_c = label(mask_c)
        cand_sizes = np.bincount(lbl_c.ravel())[1:] if num_c > 0 else []
        major_cand_comp = sum(s >= 20 for s in cand_sizes)
        real_holes_c = count_structural_holes(mask_c, min_area=15)

    # Sigmoid 抗锯齿
    ink_aligned = 1.0 / (1.0 + np.exp(-phi_prime / edge_tau))
    return (1.0 - ink_aligned).astype(np.float32)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    with open("assets/stroke_width_trainset_distribution.json", encoding="utf-8") as f:
        td = json.load(f)

    from tools.render_fixed_20row_poster import measure_p50_and_contrast
    
    # Test 1: Thin to Thick (g25, 好)
    p_thin = "assets/results/top2_u100_base_30k/g25.png"
    im_thin = np.array(Image.open(p_thin).convert("L"), dtype=np.float32) / 255.0
    p50_0, _ = measure_p50_and_contrast(im_thin)
    print(f"Thin test g25: raw p50={p50_0:.2f}px, target={td['王羲之_行']['ref_p50']:.1f}px")
    for a in [0.35, 0.70, 1.00]:
        res = test_fixed_powerful_align(im_thin, td["王羲之_行"], alpha=a)
        p50, con = measure_p50_and_contrast(res)
        print(f"  Alpha {a:.2f}: p50={p50:.2f}px (diff={p50 - p50_0:+.2f}px)")

    # Test 2: Thick to Thin (g80, 旨)
    p_thick = "artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g80.png"
    im_thick = np.array(Image.open(p_thick).convert("L"), dtype=np.float32) / 255.0
    p50_0, _ = measure_p50_and_contrast(im_thick)
    print(f"\nThick test g80: raw p50={p50_0:.2f}px, target={td['王羲之_楷']['ref_p50']:.1f}px")
    for a in [0.35, 0.70, 1.00]:
        res = test_fixed_powerful_align(im_thick, td["王羲之_楷"], alpha=a)
        p50, con = measure_p50_and_contrast(res)
        print(f"  Alpha {a:.2f}: p50={p50:.2f}px (diff={p50 - p50_0:+.2f}px)")
