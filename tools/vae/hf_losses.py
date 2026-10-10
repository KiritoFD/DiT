#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf_losses.py — VAE 修正训练用的两组新损失（高频保真 + 后验坍缩）

背景（本轮实测结论）:
  * 我们的 Calli-VAE 往返高频只有 GT 的 0.78x（SD-VAE 0.81x，FLUX-16ch 0.97x），
    而 DiT 只用到 0.51~0.67x -> 天花板与利用率都有损失。
  * 根因之一: 现有监督 = L1 + DINO(struct/style)，**没有任何一项对"高频/局部纹理"敏感**
    （L1 会抹平高频；DINO 特征被训练成对低级纹理不变；DINO-style 方差住在 34/384 维子空间）。
    —— SD-VAE 的官方配方含 LPIPS（唯一纹理敏感项），我们把它整个换掉了。
  * 根因之二: 后验没有坍缩（实测相对宽度 0.372；SD 4e-5 / FLUX 2e-4）。
    因为 L1 对 σ 的梯度期望≈0、DINO 也平滑 -> 没有任何项在推 σ→0。
    而 σ 宽 = 训练目标含 ~15% 不可预测噪声 = 下游 DiT 均值化/发糊的上游原因。

本模块提供:
  1. hf_loss(rec, gt, mode) —— 高频保真损失
       'dog'   : 高斯差分高通上的 L1（最便宜、最稳）
       'fft'   : 频谱 L1（径向等权，直接压频谱差）
       'focal' : Focal Frequency Loss (Jiang et al. 2021)，按当前差距自适应加权，专治高频缺失
  2. posterior_std_penalty(posterior) —— 显式把 σ 往 0 推（让潜码确定化, mode≡sample）
  3. clamp_logvar(posterior 的 logvar) —— 硬夹紧上限（更粗暴但最确定）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 高斯高通
def _gaussian_kernel1d(k: int, sigma: float, device, dtype):
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    w = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (w / w.sum()).view(1, 1, 1, k)


def lowpass(x, k: int = 7, sigma: float = 1.0):
    """可分离高斯低通（depthwise，按通道分组）。"""
    c = x.shape[1]
    kx = _gaussian_kernel1d(k, sigma, x.device, x.dtype)
    ky = kx.transpose(-1, -2)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), kx.expand(c, 1, 1, k), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), ky.expand(c, 1, k, 1), groups=c)
    return x


def highpass(x, k: int = 7, sigma: float = 1.0):
    return x - lowpass(x, k, sigma)


# ---------------------------------------------------------------- 频谱
def _radial_radius(h: int, w: int, device, dtype, normalize: bool = True):
    """径向频率坐标。

    ⚠ normalize=False 才是**离线普查的口径**（r = sqrt((dy/(H/2))^2 + (dx/(W/2))^2)，不再除以
      r.max()）。2026-10-10 实测: 多除一次 r.max()(=1.414) 会把 r>0.5 的阈值从 64px 推到 90.5px,
      切到另一个频段, 使 torch 版与离线普查差 41% —— 又是一个"静默口径不一致"的坑。
    """
    yy = torch.arange(h, device=device, dtype=dtype).view(-1, 1) - h / 2
    xx = torch.arange(w, device=device, dtype=dtype).view(1, -1) - w / 2
    r = torch.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
    return r / r.max() if normalize else r


def band_fraction(x, r_cut: float = 0.5):
    """★ 与离线普查【完全同口径】的指标: 灰度图 FFT 中 r>r_cut 的能量份额。

    ⚠ 2026-10-10 教训: 训练时我原本用 `|DoG高通(x)|.mean()` 的比值当 HFr, 它测的是
      **宽带幅度**(覆盖 r>0.25 以上 95%), 对"顶带份额偏低"不敏感 —— 于是出现了
      "日志 HFr≈1.03 但离线普查 0.783" 的假象, 并让我误判高频已修好。
      现在训练日志与本函数都改用与普查一致的口径。
    """
    g = x.float()
    if g.shape[1] > 1:
        g = g.mean(dim=1, keepdim=True)
    g = g - g.mean(dim=(-2, -1), keepdim=True)
    F = torch.fft.fftshift(torch.abs(torch.fft.fft2(g, norm="ortho")) ** 2, dim=(-2, -1))
    r = _radial_radius(g.shape[-2], g.shape[-1], g.device, g.dtype, normalize=False)  # ★ 普查口径
    tot = F.sum(dim=(-2, -1), keepdim=True) + 1e-12
    return (F * (r > r_cut).to(F.dtype)).sum(dim=(-2, -1), keepdim=True) / tot


def hf_loss(rec, gt, mode: str = "dog", k: int = 7, sigma: float = 1.0, r_cut: float = 0.5):
    """高频保真损失（rec/gt 均在 [-1,1]）。返回标量。

    mode:
      'topband' —— **直接优化离线普查的指标**(r>r_cut 能量份额的 L1)。
                   这是 2026-10-10 的结论所指: 之前 dog 模式把宽带幅度修到 1.0,
                   但顶带份额纹丝不动(0.783->0.788), 所以必须直接对顶带份额下手。
                   ⚠ 该指标对视网膜噪声也敏感, 必须与 L1 同时开启以免"用噪点换份额"。
      'dog'     —— 高斯差分高通上的 L1（宽带幅度; 已证不足以移动顶带）
      'fft'/'focal' —— 频谱 L1 / 自适应频谱 L1
    """
    mode = (mode or "dog").lower()
    if mode == "topband":
        return torch.nn.functional.l1_loss(band_fraction(rec, r_cut), band_fraction(gt, r_cut))

    if mode == "dog":
        return F.l1_loss(highpass(rec, k, sigma), highpass(gt, k, sigma))

    if mode in ("fft", "focal"):
        r = rec.float()
        g = gt.float()
        Fr = torch.fft.fftshift(torch.fft.fft2(r, norm="ortho"), dim=(-2, -1))
        Fg = torch.fft.fftshift(torch.fft.fft2(g, norm="ortho"), dim=(-2, -1))
        diff = torch.abs(Fr - Fg)
        if mode == "fft":
            return diff.mean()
        # Focal Frequency Loss: 按当前差距自适应加权（差距大的频率点权重更大）
        w = diff ** 2
        w = w / (w.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        return (w * (diff ** 2)).mean()

    raise ValueError(f"unknown hf mode: {mode}")


# ---------------------------------------------------------------- 后验坍缩
def posterior_std_penalty(posterior):
    """显式把 posterior.std 往 0 推（L1 对 σ 的梯度期望≈0，所以必须显式加）。"""
    sd = getattr(posterior, "std", None)
    if sd is None:
        sd = torch.exp(0.5 * posterior.logvar)
    return sd.mean()


class LogvarClamp(nn.Module):
    """硬夹紧 logvar 上限 -> σ ≤ exp(clamp/2)。clamp=-9.0 -> σ ≤ 0.011（近似确定性编码）。"""

    def __init__(self, logvar_max: float = -9.0):
        super().__init__()
        self.logvar_max = logvar_max

    def forward(self, logvar):
        return torch.clamp(logvar, max=self.logvar_max)
