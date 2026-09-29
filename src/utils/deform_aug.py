#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/utils/deform_aug.py — 基于随机连续几何形变的骨架数据增强算子 (GPU 纯显存级极速前向)"""
import torch
import torch.nn.functional as F
import numpy as np

# SD VAE 下纯白背景的均值潜变量 (4通道)
Z_BG_VEC = torch.tensor([2.18129, 1.42018, -0.00979, -1.14073]).view(1, 4, 1, 1)

def random_skeleton_deformation(g, prob=0.5, max_rot_deg=5.0, max_scale=0.08, max_shear=0.06, 
                                max_trans_px=1.5, max_elastic_px=1.5, coarse_res=6,
                                preserve_amplitude=True):
    """在 GPU 上对骨架 latent g (N, 4, 32, 32) 进行纯几何随机变形增强。
    
    1. 随机仿射: 旋转 ±rot_deg, 缩放 1±scale, 剪切 ±shear, 平移 ±trans_px
    2. 随机低频弹性畸变: 粗网格采样高斯场 + 双三次插值平滑上采样
    3. 潜变量墨迹保幅校准: 沿笔画法线方向恢复峰值能量, 根除双线性采样导致的细线淡化断裂
    """
    n, c, h, w = g.shape
    device = g.device
    dtype = g.dtype
    
    if prob <= 0:
        return g
        
    apply_mask = (torch.rand(n, 1, 1, 1, device=device) < prob)
    if not apply_mask.any():
        return g

    # 1. 随机仿射矩阵 (N, 2, 3)
    rad = np.pi / 180.0
    angles = (torch.rand(n, device=device) * 2 - 1) * (max_rot_deg * rad)
    scales_x = 1.0 + (torch.rand(n, device=device) * 2 - 1) * max_scale
    scales_y = 1.0 + (torch.rand(n, device=device) * 2 - 1) * max_scale
    shears_x = (torch.rand(n, device=device) * 2 - 1) * max_shear
    shears_y = (torch.rand(n, device=device) * 2 - 1) * max_shear
    trans_x = (torch.rand(n, device=device) * 2 - 1) * (max_trans_px / (w / 2.0))
    trans_y = (torch.rand(n, device=device) * 2 - 1) * (max_trans_px / (h / 2.0))
    
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    A = torch.zeros(n, 2, 3, device=device, dtype=torch.float32)
    A[:, 0, 0] = scales_x * cos_a + shears_x * -sin_a
    A[:, 0, 1] = scales_x * -sin_a + shears_x * cos_a
    A[:, 0, 2] = trans_x
    A[:, 1, 0] = shears_y * cos_a + scales_y * sin_a
    A[:, 1, 1] = shears_y * -sin_a + scales_y * cos_a
    A[:, 1, 2] = trans_y
    
    grid_affine = F.affine_grid(A, (n, 1, h, w), align_corners=False)
    
    # 2. 低频平滑弹性形变位移场 (N, 2, coarse_res, coarse_res) -> (N, H, W, 2)
    if max_elastic_px > 0 and coarse_res > 0:
        v_coarse = (torch.randn(n, 2, coarse_res, coarse_res, device=device) * 2 - 1) * (max_elastic_px / (w / 2.0))
        v_dense = F.interpolate(v_coarse, size=(h, w), mode='bicubic', align_corners=False)
        total_grid = grid_affine + v_dense.permute(0, 2, 3, 1)
    else:
        total_grid = grid_affine

    # 3. 墨迹保幅与网格采样
    if preserve_amplitude:
        z_bg = Z_BG_VEC.to(device=device, dtype=g.dtype)
        v = g - z_bg
        energy = v.norm(dim=1, keepdim=True)
        energy_peak = F.max_pool2d(energy, kernel_size=3, stride=1, padding=1)
        energy_peak_w = F.grid_sample(energy_peak, total_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        v_warped = F.grid_sample(v.float(), total_grid, mode='bilinear', padding_mode='border', align_corners=False).to(dtype)
        v_energy = v_warped.norm(dim=1, keepdim=True)
        
        scale = torch.clamp(energy_peak_w / torch.clamp(v_energy, min=1e-3), min=1.0, max=2.5)
        gate = torch.clamp((v_energy - 0.5) / 1.0, min=0.0, max=1.0)
        v_calib = v_warped * (1.0 + gate * (scale - 1.0))
        g_deformed = z_bg + v_calib
    else:
        g_deformed = F.grid_sample(g.float(), total_grid, mode='bicubic', padding_mode='border', align_corners=False).to(dtype)
        
    out = torch.where(apply_mask, g_deformed, g)
    return out
