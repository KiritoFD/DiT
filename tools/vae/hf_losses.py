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
def _radial_radius(h: int, w: int, device, dtype):
    yy = torch.arange(h, device=device, dtype=dtype).view(-1, 1) - h / 2
    xx = torch.arange(w, device=device, dtype=dtype).view(1, -1) - w / 2
    r = torch.sqrt(yy ** 2 + xx ** 2)
    return r / r.max()


def hf_loss(rec, gt, mode: str = "dog", k: int = 7, sigma: float = 1.0):
    """高频保真损失（rec/gt 均在 [-1,1]）。返回标量。"""
    mode = (mode or "dog").lower()
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
