import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import medial_axis, skeletonize

def smooth_antialiased_sdf_align(
    img_np: np.ndarray,
    target_dist: dict,
    alpha: float = 1.0,
    threshold: float = 0.5,
    max_shift: float = 2.0,
    edge_tau: float = 0.55
) -> np.ndarray:
    """平滑高保真抗锯齿空间变异 SDF 对齐:
    1. 保持扩散模型自然的宣纸墨晕渐变 (Sigmoid 柔化边缘, tau=0.55)
    2. 笔锋端点平滑保护 (Skeleton Endpoint Taper, 绝不断尖、绝不吃笔锋)
    3. 大核高斯空间场平滑 (sigma=3.5, 彻底消除 Voronoi 结块与阶跃瑕疵)
    """
    if abs(alpha) < 1e-4:
        return img_np

    # 1. 软墨迹提取
    ink_soft = np.clip(1.0 - img_np, 0.0, 1.0)
    ink_mask = ink_soft >= threshold
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

    scale = float(np.clip(sigma_t / sigma_g, 0.6, 1.8))
    skel_widths = 2.0 * dist_map[skel]
    target_widths = mu_t + scale * (skel_widths - mu_g)
    target_widths = np.clip(target_widths, 1.5, 40.0)

    delta_skel = (target_widths - skel_widths) / 2.0
    delta_skel = np.clip(delta_skel, -max_shift, max_shift)

    # 2. 笔锋端点平滑衰减 (若变细，对细小笔锋端点进行衰减，防止吃掉细笔尖)
    # 距离骨架边缘很近或者自身非常细 (< 3.0px) 的地方，限制负向偏移
    thin_penalty = np.clip((skel_widths - 1.5) / 2.5, 0.0, 1.0)
    delta_skel_protected = np.where(delta_skel < 0, delta_skel * thin_penalty, delta_skel)

    # 3. 空间平滑扩散 (大核高斯平滑 sigma=3.5, 避免 Voronoi 破片)
    _, (nearest_r, nearest_c) = distance_transform_edt(~skel, return_indices=True)
    delta_grid = np.zeros_like(img_np, dtype=np.float32)
    delta_grid[skel] = delta_skel_protected.astype(np.float32)
    delta_map = delta_grid[nearest_r, nearest_c]
    delta_map = gaussian_filter(delta_map, sigma=3.5)

    # 4. 连续 Signed Distance Field (SDF)
    d_in = distance_transform_edt(ink_mask)
    d_out = distance_transform_edt(~ink_mask)
    phi = d_in - d_out

    # 5. 变形场叠加
    phi_prime = phi + alpha * delta_map

    # 6. 自然宣纸墨晕渐变重建 (高保真 Sigmoid 平滑过渡)
    # 当 phi_prime == 0 时，ink 为 0.5；平滑过渡宽度约 2.5 像素
    ink_aligned = 1.0 / (1.0 + np.exp(-phi_prime / edge_tau))

    # 混合保留原本扩散模型微弱的内部浓淡纹理
    # 在墨心 (phi > 2) 区域保留原始的浓淡深浅
    core_mask = np.clip((phi_prime - 1.0) / 2.0, 0.0, 1.0)
    final_ink = (1.0 - core_mask) * ink_aligned + core_mask * np.maximum(ink_aligned, ink_soft)

    return (1.0 - final_ink).astype(np.float32)

if __name__ == "__main__":
    import json
    with open("assets/stroke_width_trainset_distribution.json", encoding="utf-8") as f:
        td = json.load(f)
    
    # Test on g91 (羲) and g88 (好)
    im_g91 = np.array(Image.open("artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g91.png").convert("L"), dtype=np.float32) / 255.0
    im_g88 = np.array(Image.open("artifacts/DiC_U100_review_20261020/images/originals/decA_70k/g88.png").convert("L"), dtype=np.float32) / 255.0
    
    target = td["王羲之_行"]
    for a in [0.0, 0.35, 0.70, 1.0]:
        res_g91 = smooth_antialiased_sdf_align(im_g91, target, alpha=a)
        res_g88 = smooth_antialiased_sdf_align(im_g88, target, alpha=a)
        Image.fromarray((res_g91 * 255).astype(np.uint8)).save(f"scratch/smooth_g91_a{int(a*100)}.png")
        Image.fromarray((res_g88 * 255).astype(np.uint8)).save(f"scratch/smooth_g88_a{int(a*100)}.png")
    print("Smooth antialiased alignment generated successfully!")
