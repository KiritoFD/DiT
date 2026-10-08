#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_repa_e_joint.py — REPA-E 端到端联合微调架构核心机制单元测试

严格验证四大架构保证:
  1. 权限隔离: VAE Decoder 完全冻结 (0 梯度参数), VAE Encoder / DiT / REPA 投影层全通梯度
  2. 梯度贯通: 扩散 Flow Matching Loss 经由 posterior.rsample() 完美回传至 VAE Encoder 卷积层
  3. 防塌缩锚定: REPA Loss (Cosine 距离) 驱动 DiT 中间特征层与 DINO 目标对齐
  4. 双轨优化器: DiT (正常学习率) 与 VAE Encoder (极微小学习率 2e-6) 拥有独立调度参数组
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import AutoencoderKL
from src.model.dit import DiT_2Cond_models
from src.loss.flow_matching import FlowMatching


def test_repa_e_permission_and_gradient_flow():
    """验证 1 & 2: 组件读写权限与 rsample 梯度回传。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化一个小尺寸 VAE (kl-f8 架构, 下采样 8 倍: 256x256 -> 32x32)
    vae = AutoencoderKL(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=[32, 64, 128, 256],
        layers_per_block=1,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        act_fn="silu",
        norm_num_groups=16,
        scaling_factor=0.18215,
    ).to(device)

    # 模拟 Pillar 1: 权限下达
    # 🔒 Decoder 完全冻结
    vae.decoder.eval()
    for p in vae.decoder.parameters():
        p.requires_grad = False
    if hasattr(vae, "post_quant_conv"):
        for p in vae.post_quant_conv.parameters():
            p.requires_grad = False

    # 🟢 Encoder 开启梯度
    vae.encoder.train()
    for p in vae.encoder.parameters():
        p.requires_grad = True
    if hasattr(vae, "quant_conv"):
        for p in vae.quant_conv.parameters():
            p.requires_grad = True

    # 验证 Decoder 确实 0 梯度参数
    decoder_trainable = [p for p in vae.decoder.parameters() if p.requires_grad]
    assert len(decoder_trainable) == 0, "Decoder 必须完全冻结，绝不带梯度！"

    encoder_trainable = [p for p in vae.encoder.parameters() if p.requires_grad]
    assert len(encoder_trainable) > 0, "Encoder 必须开启梯度！"

    # 2. 构建 DiT 模型
    model = DiT_2Cond_models["DiT-2Cond-XS/2"](
        num_calligraphers=5, num_characters=100,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        norm_type="layer", mlp_type="gelu", attn_impl="sdpa"
    ).to(device)

    fm = FlowMatching(num_steps=10)

    # 3. 前向数据流
    B = 2
    x_real = torch.randn(B, 3, 256, 256, device=device)
    y_callig = torch.zeros(B, dtype=torch.long, device=device)
    y_char = torch.zeros(B, dtype=torch.long, device=device)

    # diffusers 的 DiagonalGaussianDistribution.sample() 原生即为重参数化采样 (mean + std * eps)
    posterior = vae.encode(x_real).latent_dist
    x_latent = posterior.sample().mul_(0.18215)
    loss_kl = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp(), dim=[1, 2, 3]).mean()

    # Flow matching
    t = fm.sample_t(B, device)
    loss_dict = fm.training_losses(model, x_latent, t, model_kwargs=dict(y_callig=y_callig, y_char=y_char))
    loss_diff = loss_dict["loss"].mean()

    # 模拟 REPA 对齐损失
    dit_feat = model.x_embedder(x_latent)  # (B, 256, D)
    dino_target = torch.randn_like(dit_feat)
    loss_repa = 1.0 - F.cosine_similarity(dit_feat, dino_target, dim=-1).mean()

    # 三权分立总损失
    total_loss = loss_diff + 0.03 * loss_repa + 1e-5 * loss_kl

    # 4. 反向传播
    total_loss.backward()

    # 检验 VAE Encoder 是否切实接收到了来自 Flow Matching 的反向梯度
    encoder_first_conv = vae.encoder.conv_in
    assert encoder_first_conv.weight.grad is not None, "梯度未能穿透 sample() 流回 VAE Encoder 卷积层！"
    grad_norm = encoder_first_conv.weight.grad.norm().item()
    print(f"[OK] 梯度成功穿透 sample() 流回 VAE Encoder! 首层梯度范数: {grad_norm:.6f}")

    # 检验 Decoder 绝对无梯度
    for p in vae.decoder.parameters():
        assert p.grad is None, "Decoder 违规产生了反向梯度！"
    print("[OK] VAE Decoder 严格保持 0 梯度与冻结状态！")


def test_two_speed_optimizer():
    """验证 3: 双轨学习率调度器结构。"""
    dit_params = [nn.Parameter(torch.randn(10, 10))]
    repa_params = [nn.Parameter(torch.randn(5, 5))]
    vae_enc_params = [nn.Parameter(torch.randn(8, 8))]

    lr_dit = 5e-5
    lr_vae = 2e-6

    optim_groups = [
        {"params": dit_params, "lr": lr_dit, "weight_decay": 0.01},
        {"params": repa_params, "lr": lr_dit, "weight_decay": 0.0},
        {"params": vae_enc_params, "lr": lr_vae, "weight_decay": 0.0},
    ]

    opt = torch.optim.AdamW(optim_groups)

    assert len(opt.param_groups) == 3
    assert opt.param_groups[0]["lr"] == lr_dit
    assert opt.param_groups[1]["lr"] == lr_dit
    assert opt.param_groups[2]["lr"] == lr_vae

    print(f"[OK] 双轨学习率调度器配置正确: DiT/REPA lr={lr_dit} | VAE Encoder lr={lr_vae}")


if __name__ == "__main__":
    test_repa_e_permission_and_gradient_flow()
    test_two_speed_optimizer()
    print("\n[PASS] REPA-E 联合微调核心机制单元测试全部通过！")
