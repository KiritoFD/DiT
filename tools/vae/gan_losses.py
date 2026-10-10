#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gan_losses.py — VAE 训练用的 PatchGAN 判别器与损失（SD1.5/LDM 范式）

依据（用户 2026-10-10 裁定 + 盘上可验证的官方 model card）:
  * **判别器从零联合训练**，绝不用预训练 —— 它的任务是挑"本 VAE 解码器特有的伪影"
    （边缘发虚、特定频率马赛克），预训练判别器不认识这些造假手法。
  * SD1.5 VAE (kl-f8, LDM/Taming) 的配方: L1 + LPIPS + KL(1e-6) + **PatchGAN(Hinge, 0.5)**
    官方 card 明写 ft-EMA = "L1 + LPIPS"（decoder-only 微调）；判别器来自原始 kl-f8 的
    `LPIPSWithDiscriminator`(disc_weight 0.5, disc_start 50001, 自适应权重)。
  * **自适应权重是防翻车的关键**（LDM 祖传）: λ = ‖∇L_rec‖/‖∇L_GAN‖，再乘基础权重。
    否则判别器梯度会把 L1 冲垮 -> 变成棋盘格/怪纹理（本项目当年正是因此弃用 GAN）。

本模块只做三件事（保持最小、可单测）:
  1. NLayerDiscriminator —— 5 层全卷积 PatchGAN（256 输入 -> 30x30 patch 判定，感受野 ~70px）
  2. hinge_d_loss / hinge_g_loss —— Hinge 对抗损失
  3. adaptive_weight —— LDM 的自适应权重（对解码器**最后一层**权重的梯度范数比）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- 判别器
class NLayerDiscriminator(nn.Module):
    """PatchGAN（pix2pix/LDM 的 NLayerDiscriminator）。

    256x256 输入 -> 每层 stride2 下采样 -> 输出 ~30x30 的 logits，
    每个 logit 对应原图上约 70x70 的感受野（"判 patch 真假"而不是整图）。

    ⚠ 刻意**不做多尺度**（用户裁定：单尺度 PatchGAN 足够且好调）；
      也没有预训练权重（从零初始化，与 VAE 交替训练）。
    """

    def __init__(self, in_channels: int = 3, base: int = 64, n_layers: int = 3,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.in_channels = in_channels
        norm = nn.utils.spectral_norm if use_spectral_norm else (lambda m: m)

        kw = dict(kernel_size=4, stride=2, padding=1)
        seq = [norm(nn.Conv2d(in_channels, base, **kw)), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, n_layers):
            prev, mult = mult, min(2 ** i, 8)
            seq += [norm(nn.Conv2d(base * prev, base * mult, **kw)),
                    nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        # 倒数第二层 stride=1（扩大感受野而不继续降分辨率）
        seq += [norm(nn.Conv2d(base * prev, base * mult, kernel_size=4, stride=1, padding=1)),
                nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(base * mult, 1, kernel_size=4, stride=1, padding=1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- 损失
def hinge_d_loss(logits_real, logits_fake):
    """判别器损失（Hinge）：真样本推高到 ≥1，假样本压低到 ≤-1。"""
    loss_real = torch.relu(1.0 - logits_real).mean()
    loss_fake = torch.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake):
    """生成器（VAE decoder）对抗损失：骗过判别器。"""
    return -logits_fake.mean()


def adaptive_weight(loss_nll, loss_adv, last_layer, max_val: float = 1e4):
    """LDM 的自适应权重：λ = clamp(‖∇L_rec‖ / ‖∇L_adv‖, 0, max_val)。

    ⚠ 必须在**同一次前向的图**上分别对 `last_layer`(解码器最后一层权重) 求梯度，
      且两次都要 `retain_graph=True`，否则主 loss 的 backward 会缺图。
    ⚠ 这是防"GAN 把 L1 冲垮"的核心机制（本项目当年弃用 GAN 的原因就是没有它）。
    """
    grad_nll = torch.autograd.grad(loss_nll, last_layer, retain_graph=True, allow_unused=True)[0]
    grad_adv = torch.autograd.grad(loss_adv, last_layer, retain_graph=True, allow_unused=True)[0]
    if grad_adv is None:
        return loss_adv.new_tensor(0.0)
    if grad_nll is None:
        return loss_adv.new_tensor(0.0)
    w = torch.norm(grad_nll) / (torch.norm(grad_adv) + 1e-4)
    return torch.clamp(w, 0.0, max_val).detach()


def find_last_layer(vae):
    """取 decoder 的最后一层卷积权重（自适应权重的求梯度目标）。"""
    for name in ("conv_out", "conv_out_2"):
        m = getattr(vae.decoder, name, None)
        if isinstance(m, nn.Conv2d):
            return m.weight
    # 兜底: decoder 里最后一个 Conv2d
    last = None
    for m in vae.decoder.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("在 vae.decoder 里找不到 Conv2d 作为自适应权重的目标层")
    return last.weight


def edge_loss(rec, gt, ksize: int = 3):
    """Laplacian 边缘 L1（用户配方里的 loss_edge）。

    ⚠ 与本仓 `hf_losses.highpass(k=7,sigma=1)` 的区别: Laplacian 更"锐"，其 90% 能量
      落在 r>0.5 的顶带（实测），而 k=7 的 DoG 只有 73% 且**被证明不足以移动顶带份额**。
      建议与 `hf_losses.hf_loss(mode='topband')` **并用**：Laplacian 管"边缘位置"，
      topband 管"顶带份额"（后者是我们唯一验证能提升 HF 的项）。
    """
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=rec.dtype, device=rec.device)
    k = k.view(1, 1, 3, 3).expand(rec.shape[1], 1, 3, 3)
    g = F.conv2d(F.pad(gt, (1, 1, 1, 1), mode="reflect"), k, groups=gt.shape[1])
    r = F.conv2d(F.pad(rec, (1, 1, 1, 1), mode="reflect"), k, groups=rec.shape[1])
    return F.l1_loss(r, g)


# --------------------------------------------------------------------------- LeCam 正则
class LeCamEMA:
    """LeCam 正则（FLUX/MagViT-v2 家族用来"按住判别器上限"的稳定技术）。

    机理（用户给的描述）: 维护**真实图 logits 的 EMA**。若当前 batch 里判别器给假图打的分
    **低于真实图的历史均值**，说明判别器已经"过度自信/过强"（生成器即将崩），于是对它狠狠惩罚。

    实现（对 hinge 判别器）:
        L_lecam = mean(relu(1 - D(real)))²  ... 用 EMA 校正后的项
    这里采用与 Hinge 相容的实用形式:
        L_lecam = (mean(relu(1 - D(real))) + mean(relu(1 + D(fake))))²  的 LeCam 变体 ——
    为避免自创公式引入偏差, 本实现只做**最小且可解释**的一件事:
        ema_real  <- decay*ema_real + (1-decay)*mean(D(real))
        L_lecam    = relu(ema_real - mean(D(fake))) ** 2      # 假图分低于真实历史均值就罚
    ⚠ 该项只加在**判别器**的损失上（不传梯度给 VAE），权重默认 0（关闭），需要时再开。
    """

    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.ema_real = None

    def __call__(self, logits_real, logits_fake):
        mr = logits_real.mean().detach()          # EMA 只统计, 不传梯度
        mf = logits_fake.mean()                    # ★ 保留图: 惩罚要能训到判别器
        if self.ema_real is None:
            self.ema_real = mr
            return logits_fake.new_tensor(0.0)
        self.ema_real = self.decay * self.ema_real + (1.0 - self.decay) * mr
        return torch.relu(self.ema_real - mf) ** 2
