#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_warp.py — 变形算子本身会不会把笔画弄断?

隔离变量: 不训练、不涉及损失。只做
    编码(w px 骨架) -> grid_sample 位移 -> 解码 -> 量墨量/连通分量
对比:
  · 不同笔宽 w (3 / 7 / 20 px): 粗笔画是否更抗 warp?
  · 不同 warp_iters (1 / 4): 级联小位移是否真的更保形?
  · 不同位移幅度 mag (0 / 0.5 / 1 / 2 / 4 格)

判据: 解码墨量比 (相对未形变) 与 连通分量比。1.0 = 算子无损。
"""
import glob
import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '/root/Workspace/xy/DiT')

from diffusers.models import AutoencoderKL
from scipy.ndimage import binary_dilation, label
from skimage.morphology import skeletonize

DEV = 'cuda'
SF = 0.18215
GRID = 32

vae = AutoencoderKL.from_pretrained(
    'data/pretrained/pretrained_models/sd-vae-ft-ema').to(DEV).eval()
for p in vae.parameters():
    p.requires_grad_(False)

pngs = [p for p in sorted(glob.glob('data/top10_style23/std/*.png'))][:64]
print(f'std 骨架 {len(pngs)} 张')


def to_tensor(ink):
    a2 = np.where(ink, 0.0, 1.0).astype(np.float32)
    t = th.from_numpy(a2)[None, None].repeat(1, 3, 1, 1)
    t = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)
    return t * 2.0 - 1.0


@th.no_grad()
def encode(ts):
    out = []
    for i in range(0, len(ts), 32):
        x = th.cat(ts[i:i + 32]).to(DEV)
        out.append(vae.encode(x).latent_dist.mode() * SF)
    return th.cat(out, 0).float()


@th.no_grad()
def decode(z):
    return vae.decode(z / SF).sample.float()


def stats(gray, thr=0.5):
    """gray (B,3,256,256) in [-1,1] -> (墨量占比按0阈值, 平均连通分量数)"""
    ink = (gray.mean(1) < 0).cpu().numpy()      # [-1,1]: <0 表示偏暗 = 墨
    struct = np.ones((3, 3), int)
    mass, ncomp = [], []
    for b in ink:
        mass.append(float(b.mean()))
        _, n = label(b, structure=struct)
        ncomp.append(n)
    return float(np.mean(mass)), float(np.mean(ncomp))


def stats_shape(gray, q=0.05):
    """幅度无关的形状指标: 每张图取自己最暗的 q 比例像素当墨, 只量连通性。

    ⚠ 为什么必须这样: `gray < 0` 对**幅度**极敏感 —— 上采样/warp/降采样会轻微
      糊掉峰值, 阈值一挪墨量就掉一个数量级, 于是"算子变差"其实是"峰值变低"。
      固定分位数把幅度因素除掉, 剩下的才是笔画有没有断。
    """
    g = gray.mean(1).cpu()
    B = g.shape[0]
    flat = g.flatten(1)
    thr = flat.kthvalue(max(1, int(q * flat.shape[1])), dim=1).values
    ink = (flat <= thr.view(B, 1)).view(g.shape).numpy()
    struct = np.ones((3, 3), int)
    ncomp = []
    for b in ink:
        _, n = label(b, structure=struct)
        ncomp.append(n)
    return float(np.mean(ncomp))


idx = (2.0 * (th.arange(GRID) + 0.5) / GRID) - 1.0
gy, gx = th.meshgrid(idx, idx, indexing='ij')
BASE = th.stack([gx, gy], -1).unsqueeze(0).to(DEV)         # (1,32,32,2)

# 墨迹方向 (与 deform_skel 的 delta_ink 同口径), 用于"保质量 warp"
Z_BG = th.tensor([2.18129, 1.42018, -0.00979, -1.14073],
                 device=DEV).view(1, 4, 1, 1)
D_DIR = th.tensor([-3.16140, -4.00872, 1.12232, 2.44625],
                  device=DEV).view(1, 4, 1, 1)
D_DIR = D_DIR / D_DIR.norm()


def ink_mass(z):
    """沿墨迹方向的总墨量 (B,)"""
    return (((z - Z_BG) * D_DIR).sum(1)).sum(dim=(1, 2))


def mass_normalize(z_out, z_in):
    """闭式保质量: 沿墨迹方向均匀补偿, 使总墨量与输入一致。

    ip 对 z 是线性的: 给 z 加 c·D_DIR 会让每个像素的 ip 精确增加 c。
    所以补 (M_in - M_out)/N 即可, 无需迭代、无参数。
    """
    n = z_out.shape[-1] * z_out.shape[-2]
    d = (ink_mass(z_in) - ink_mass(z_out)) / n
    return z_out + d.view(-1, 1, 1, 1) * D_DIR


def smooth_field(n, mag, seed):
    g = th.Generator(device='cpu').manual_seed(seed)
    lo = th.randn(n, 2, 8, 8, generator=g)
    up = F.interpolate(lo, size=(GRID, GRID), mode='bilinear',
                       align_corners=False)
    up = up.permute(0, 2, 3, 1)
    m = up.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (up / m * mag).to(DEV)


def warp_at(Z, fld, scale=1, iters=1):
    """在 scale 倍分辨率上做重采样再降回 32x32。

    动机: 32x32 上一根 3px 骨架只有 0.375 格, 双线性重采样必然把它抹平。
    上采样到 H=32*scale 后同一根线占 0.375*scale 格, 重采样损伤随分辨率下降;
    位移用**归一化坐标**表达(与分辨率无关), 所以物理位移不变。
    """
    if scale == 1:
        Zh, H = Z, GRID
        base = BASE
        fH = fld
    else:
        H = GRID * scale
        Zh = F.interpolate(Z, scale_factor=scale, mode='bilinear',
                           align_corners=False)
        idxh = (2.0 * (th.arange(H, device=DEV) + 0.5) / H) - 1.0
        gyh, gxh = th.meshgrid(idxh, idxh, indexing='ij')
        base = th.stack([gxh, gyh], -1).unsqueeze(0)
        fH = F.interpolate(fld.permute(0, 3, 1, 2), size=(H, H),
                           mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    out = Zh
    for _ in range(iters):
        gk = base + fH / (GRID / 2.0) / iters
        out = F.grid_sample(out, gk, mode='bilinear',
                            padding_mode='border', align_corners=False)
    if scale > 1:
        out = F.interpolate(out, size=(GRID, GRID), mode='area')
    return out


print()
print(f"{'w':>4} {'mag':>5} {'算子':>8} {'墨量比':>8} {'连通比':>8} {'形状连通比':>11} {'未形变连通':>10}")
for w in (3, 7, 20):
    it = max(0, (w - 1) // 2)
    inks = []
    for p in pngs:
        b = np.asarray(Image.open(p).convert('L')) < 128
        if b.sum() > 0:
            b = skeletonize(b)
        inks.append(binary_dilation(b, iterations=it) if it > 0 else b)
    Z = encode([to_tensor(b) for b in inks])
    m0, c0 = stats(decode(Z))
    s0 = stats_shape(decode(Z))
    for mag in (1.0, 2.0, 4.0):
        fld = smooth_field(Z.shape[0], mag, seed=0)
        for tag, sc, itn in (('plain', 1, 1), ('hi4', 4, 1), ('hi8', 8, 1)):
            Zm = warp_at(Z, fld, scale=sc, iters=itn)
            m1, c1 = stats(decode(Zm))
            s1 = stats_shape(decode(Zm))
            print(f'{w:>4} {mag:>5.1f} {tag:>8} {m1 / max(m0, 1e-9):>8.3f} '
                  f'{c1 / max(c0, 1e-9):>8.3f} {s1 / max(s0, 1e-9):>11.3f} '
                  f'{c0:>7.1f}/{s0:.1f}')
    print()
