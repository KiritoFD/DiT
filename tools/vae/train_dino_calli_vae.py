#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_dino_calli_vae.py — 基于纯 DINOv2 感知与笔法方差监督的 Calli-VAE 全量微调器

核心创新设计 (零对抗、高保真、彻底终结 VAE 高斯平滑与飞白丢失):
  1. 基础重建与高斯保底:
     loss_l1 = F.l1_loss(x_recon, x_real)
     loss_kl = KL(posterior)
  2. 结构保真度损失 (DINO-REPA 结构损失):
     feat_real = DINOv2(x_real), feat_recon = DINOv2(x_recon)
     loss_dino_struct = F.mse_loss(feat_recon, feat_real)  # 替代传统 VGG-LPIPS
  3. 【核心大杀器】DINO 笔法方差损失 (替代不稳定的 GAN 判别器):
     std_real = feat_real.std(dim=1), std_recon = feat_recon.std(dim=1)
     loss_dino_style = F.mse_loss(std_recon, std_real)    # 强迫保留干枯飞白与高频笔锋！
"""

import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

from diffusers import AutoencoderKL
from safetensors import safe_open
from transformers import Dinov2Config, Dinov2Model


class CalligraphyDataset(Dataset):
    """读取 39.3 万真实书法图像数据集。"""
    def __init__(self, csv_path, data_root, size=256, max_samples=None):
        self.data_root = data_root
        self.size = size
        self.items = []

        print(f"[data] 读取数据集索引: {csv_path} ...", flush=True)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get("image_path") or row.get("path")
                if not rel_path:
                    img_id = row.get("img_id") or row.get("id")
                    rel_path = f"imgs/{img_id}.png"
                self.items.append(rel_path)
                if max_samples and len(self.items) >= max_samples:
                    break

        print(f"[data] 成功载入 {len(self.items)} 条样本！", flush=True)

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),             # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_p = self.items[idx]
        full_p = os.path.join(self.data_root, rel_p)
        try:
            img = Image.open(full_p).convert("RGB")
            t = self.transform(img)
            return t
        except Exception:
            # 容错降级: 若偶发文件读取异常，回退纯白底张量 ([-1, 1])
            return torch.ones((3, self.size, self.size), dtype=torch.float32)


def load_dino_model(ckpt_path, device):
    """从本地 safetensors 加载 DINOv2-S/14 特征提取器 (开启 SDPA 极速注意力和零显存冗余)。"""
    print(f"[dino] 从本地权重载入 DINOv2: {ckpt_path} ...", flush=True)
    with safe_open(ckpt_path, framework="pt") as f:
        keys = list(f.keys())
        hidden = f.get_tensor("encoder.layer.0.attention.attention.query.weight").shape[0]
    depth = max(int(k.split(".")[2]) for k in keys if k.startswith("encoder.layer.")) + 1
    heads = {384: 6, 768: 12, 1024: 16, 1536: 24}.get(hidden, 6)
    n_reg = 1 if any("register_tokens" in k for k in keys) else 0

    config = Dinov2Config(
        image_size=224,
        patch_size=14,
        num_channels=3,
        hidden_size=hidden,
        num_hidden_layers=depth,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_scale_init_value=1.0,
        num_register_tokens=n_reg,
    )
    try:
        model = Dinov2Model(config, attn_implementation="sdpa")
        print("  [dino] 成功激活 SDPA 极速注意力核心！", flush=True)
    except Exception as e:
        print(f"  [dino] SDPA 不可用 ({e})，回退默认实现", flush=True)
        model = Dinov2Model(config)

    # ★ [修正 2026-10-09] transformers 命名差异: 4.x 用 attention.attention.query，
    #   5.x 用 attention.q_proj。**必须按目标模型自己的 state_dict 键决定是否改名** ——
    #   旧代码无条件改成 5.x 命名，在 4.x 环境 (如 4090 的 transformers 4.36.2) 会让
    #   主干权重全部 miss、整网随机初始化，而 strict=False 只打印一行 warning，
    #   训练照跑，DINO 感知损失彻底失效 (静默错误)。
    model_keys = set(model.state_dict().keys())
    use_new_attn_names = any(".attention.q_proj." in k for k in model_keys)
    print(f"  [dino] 目标模型注意力命名: {'q_proj (transformers>=5)' if use_new_attn_names else 'attention.query (transformers 4.x)'}", flush=True)

    sd = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            new_k = k
            if use_new_attn_names:
                new_k = new_k.replace("attention.attention.query", "attention.q_proj")
                new_k = new_k.replace("attention.attention.key", "attention.k_proj")
                new_k = new_k.replace("attention.attention.value", "attention.v_proj")
                new_k = new_k.replace("attention.output.dense", "attention.o_proj")
            sd[new_k] = v

    # 动态重采样位置编码: 518x518 (1370 tokens) -> 224x224 (257 tokens)
    pe = sd.get("embeddings.position_embeddings")
    target_patches = (config.image_size // config.patch_size) ** 2  # 16 * 16 = 256
    if pe is not None and pe.shape[1] != 1 + target_patches:
        cls_pe = pe[:, :1]
        patch_pe = pe[:, 1:]
        h = w = int(round((patch_pe.shape[1]) ** 0.5))
        patch_pe = patch_pe.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(config.image_size // config.patch_size,
                                                 config.image_size // config.patch_size),
                                 mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, target_patches, -1)
        sd["embeddings.position_embeddings"] = torch.cat([cls_pe, patch_pe], dim=1)
        print(f"  [dino] 成功插值对齐 position_embeddings: {tuple(pe.shape)} -> {tuple(sd['embeddings.position_embeddings'].shape)}", flush=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [dino warn] missing keys: {missing[:5]}", flush=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class DINOPerceptualLoss(nn.Module):
    """纯 DINOv2 驱动的结构对齐与笔法方差感知损失。"""
    def __init__(self, dino_model, device):
        super().__init__()
        self.dino = dino_model
        # ImageNet 归一化常量
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _preprocess(self, x):
        """输入 [-1, 1] 256x256 -> 转为 [0, 1] -> 双三次插值到 224x224 -> ImageNet 归一化。"""
        x_01 = (x + 1.0) / 2.0
        x_224 = F.interpolate(x_01, size=(224, 224), mode="bicubic", align_corners=False)
        return (x_224 - self.mean) / self.std

    def extract_patch_features(self, x):
        """提取 (B, 256, 384) 的 Patch Tokens 特征矩阵。"""
        x_norm = self._preprocess(x)
        outputs = self.dino(x_norm)
        # 去掉首位的 CLS Token，保留 256 个空间 Patch Tokens
        return outputs.last_hidden_state[:, 1:, :]

    def forward(self, x_real, x_recon):
        # 1. 真实图提取 (不计梯度)
        with torch.no_grad():
            feat_real = self.extract_patch_features(x_real)  # (B, N, D)
            std_real = feat_real.std(dim=1)                  # (B, D)

        # 2. 重建图提取 (梯度回传穿透整个 VAE)
        feat_recon = self.extract_patch_features(x_recon)    # (B, N, D)
        std_recon = feat_recon.std(dim=1)                    # (B, D)

        # 3. 结构保真损失 (Patch 级拓扑对齐)
        loss_struct = F.mse_loss(feat_recon, feat_real)

        # 4. 笔法方差损失 (强迫恢复笔墨浓淡与飞白多样性)
        loss_style = F.mse_loss(std_recon, std_real)

        return loss_struct, loss_style


def verify_decode_convention(vae, x, device, sf):
    """实测该 VAE 的正确解码输入, 并对照我们训练用的 decode(sample)。

    背景 (2026-10-09): 旧版训练脚本写 `x_recon = vae.decode(z / sf)`，等于把 decoder
    的输入尺度放大 1/sf ≈ 5.49 倍。实测已证明它训出一个"只认 sample/sf"的**非标准**
    解码器 (base sd-vae 是 sample→L1 0.011 / sample/sf→0.417；旧 calli 恰好相反：
    sample→0.848 / sample/sf→0.038)。下游一旦按标准约定 decode(pred/sf)=decode(mode)
    去解，就得到灰图。此自检把"约定"从口头约定变成开机实测，防止再次静默跑歪。
    """
    with torch.no_grad():
        post = vae.encode(x).latent_dist
        s, m = post.sample(), post.mode()
        l1_s = F.l1_loss(vae.decode(s).sample, x).item()
        l1_ssf = F.l1_loss(vae.decode(s / sf).sample, x).item()
        l1_m = F.l1_loss(vae.decode(m).sample, x).item()
        l1_msf = F.l1_loss(vae.decode(m / sf).sample, x).item()
    std = post.std.mean().item()
    gap = (s - m).abs().mean().item()
    print("[conv-selfcheck] 解码输入 -> 重建 L1 (越低越好):", flush=True)
    print(f"    decode(sample)     = {l1_s:.4f}", flush=True)
    print(f"    decode(sample/sf)  = {l1_ssf:.4f}", flush=True)
    print(f"    decode(mode)       = {l1_m:.4f}", flush=True)
    print(f"    decode(mode/sf)    = {l1_msf:.4f}", flush=True)
    print(f"[conv-selfcheck] posterior.std={std:.4f}  |sample-mode|={gap:.4f}", flush=True)

    # 判定的是"缩放约定" (真正的 bug 类别)，而非 sample-vs-mode 的 argmin：
    # base sd-vae-ft-ema 后验极窄 (std≈2e-4)，sample 与 mode 打平，用 argmin 会误判。
    if l1_s <= l1_ssf * 0.5:
        verdict, ok = "STANDARD —— decoder 吃原始 z: decode(sample)  ✓", True
    elif l1_ssf <= l1_s * 0.5:
        verdict, ok = ("INVERTED —— decoder 只认 sample/sf，典型的 decode(z/sf) 训练残留  ✗", False)
    else:
        verdict, ok = ("SCALE-INSENSITIVE —— 后验极窄(sample≈mode) 且 sf 影响很小；"
                       "标准 decode(sample) 即可  ✓", True)
    print(f"[conv-selfcheck] 约定判定: {verdict}", flush=True)
    print(f"[conv-selfcheck] 本脚本训练约定 = decode(posterior.sample())  (与标准一致: {ok})", flush=True)
    if not ok:
        print("[conv-selfcheck] ⚠ base 已非标准。本脚本仍按标准 decode(sample) 训练; "
              "完训后会再自检, 确认其已回到 decode(sample)。", flush=True)
    return ok


def kl_divergence(posterior):
    """标准 KL 散度: D_KL(q(z|x) || N(0, I))"""
    return -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp(), dim=[1, 2, 3]).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Calli-VAE DINO-REPA 纯监督微调")
    parser.add_argument("--vae-base", type=str,
                        default="/home/ds/Workspace/moyi/models/sd-vae-ft-ema",
                        help="预训练基线 VAE 目录")
    parser.add_argument("--dino-ckpt", type=str,
                        default="/home/ds/Workspace/DiT/pretrained_models/dinov2_vits14_pretrain.safetensors",
                        help="DINOv2 本地 safetensors 权重路径")
    parser.add_argument("--csv", type=str,
                        default="/home/ds/Workspace/moyi/data/unified_393k/train_clean.csv",
                        help="40万样本元数据 CSV")
    parser.add_argument("--data-root", type=str,
                        default="/home/ds/Workspace/moyi/data/unified_393k",
                        help="图像文件所在根目录")
    parser.add_argument("--output", type=str,
                        default="/home/ds/Workspace/DiT/experiments/calli_vae_dino",
                        help="权重与可视化输出目录")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本限制 (None=全量 39.3 万)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="单步微批次大小 (推荐 16, 配合 --grad-accum 4 达成等效 Batch 64)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="梯度累积步数 (等效 Batch = batch-size * grad-accum)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="是否开启 VAE 激活检查点 (默认关闭，避免额外重计算损耗算力)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=None,
                        help="余弦学习率下限 (默认 = 0.1 * lr)")
    parser.add_argument("--w-l1", type=float, default=1.0,
                        help="L1 重建损失权重")
    parser.add_argument("--w-kl", type=float, default=1e-4,
                        help="KL 散度正则化权重")
    parser.add_argument("--w-struct", type=float, default=0.1,
                        help="DINO 结构保真损失权重")
    parser.add_argument("--w-style", type=float, default=0.5,
                        help="DINO 笔法方差损失权重 (核心飞白保障项)")
    parser.add_argument("--max-steps", type=int, default=30000,
                        help="总训练步数 (30k 步约覆盖 200 万次样本前向)")
    parser.add_argument("--save-every", type=int, default=2500,
                        help="保存权重频率")
    parser.add_argument("--vis-every", type=int, default=500,
                        help="生成对比海报频率")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def save_visual_grid(x_real, x_recon, out_path, num_show=8):
    """保存真实图 vs 重建图对比栅格。"""
    num_show = min(num_show, x_real.shape[0])
    canvas_w = num_show * 256
    canvas_h = 2 * 256
    poster = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    x_real_np = ((x_real[:num_show].float().clamp(-1, 1).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    x_recon_np = ((x_recon[:num_show].float().clamp(-1, 1).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    for i in range(num_show):
        im_real = Image.fromarray(x_real_np[i])
        im_recon = Image.fromarray(x_recon_np[i])
        poster.paste(im_real, (i * 256, 0))
        poster.paste(im_recon, (i * 256, 256))

    poster.save(out_path)


def main():
    args = parse_args()
    eta_min = args.eta_min if args.eta_min is not None else args.lr * 0.1
    os.makedirs(args.output, exist_ok=True)
    vis_dir = os.path.join(args.output, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("【启动纯 DINO-REPA 监督 Calli-VAE 全量微调】")
    print(f"  基线 VAE: {args.vae_base}")
    print(f"  DINO 裁判: {args.dino_ckpt}")
    print(f"  训练数据: {args.csv}")
    print(f"  Batch: {args.batch_size} | 学习率: {args.lr} (eta_min={eta_min}) | 目标步数: {args.max_steps}")
    print(f"  损失配方: {args.w_l1}*L1 + {args.w_kl}*KL + {args.w_struct}*DINO_Struct + {args.w_style}*DINO_Style")
    print(f"  输出目录: {args.output}")
    print("=" * 80)

    # 1. 载入 DINOv2 裁判模型
    dino_raw = load_dino_model(args.dino_ckpt, device)
    dino_loss_fn = DINOPerceptualLoss(dino_raw, device)

    # 2. 载入基线 VAE 并解冻全量参数 (Encoder + Decoder)
    print(f"[vae] 从 {args.vae_base} 载入 AutoencoderKL ...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_base).to(device)
    if args.gradient_checkpointing and hasattr(vae, "enable_gradient_checkpointing"):
        try:
            vae.enable_gradient_checkpointing()
            print("  [vae] 成功激活 VAE 梯度检查点 (Gradient Checkpointing)！节省约 60% 激活显存。", flush=True)
        except Exception as e:
            print(f"  [vae warn] 启用梯度检查点失败 ({e})，继续以常规模式运行", flush=True)

    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    print(f"  [vae] 全量可训参数量: {sum(p.numel() for p in trainable_params)/1e6:.2f}M (Encoder + Decoder)", flush=True)

    # 3. 构造数据集与 DataLoader
    dataset = CalligraphyDataset(args.csv, args.data_root, size=256, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 4. 优化器与余弦学习率调度器
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=eta_min)

    # 固定的 8 张验证图 (固定观察同一批字的细节演化)
    fixed_val_batch = next(iter(dataloader))[:8].to(device)

    # ★ [2026-10-09] 开机约定自检: 把"latent 喂 sample 还是 sample/sf"从口头约定
    #   变成实测证据 (正是旧版 decode(z/sf) 静默跑歪的地方, 详见 verify_decode_convention)。
    verify_decode_convention(vae, fixed_val_batch, device, vae.config.scaling_factor)

    step = 0
    micro_step = 0
    epoch = 0
    log_path = os.path.join(args.output, "train_dino_vae.log")
    log_file = open(log_path, "a", encoding="utf-8")

    t_start = time.time()
    accum_l1 = 0.0
    accum_struct = 0.0
    accum_style = 0.0
    accum_kl = 0.0
    accum_total = 0.0

    print(f"\n>>> 开始全速迭代训练 (微批次={args.batch_size}, 累积步数={args.grad_accum}, 等效Batch={args.batch_size * args.grad_accum}) ...", flush=True)

    optimizer.zero_grad()

    while step < args.max_steps:
        epoch += 1
        for x_real in dataloader:
            if step >= args.max_steps:
                break
            micro_step += 1
            x_real = x_real.to(device)

            # 前向编解码与重参数化采样
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # ↑ [修正 2026-10-09] 旧代码是 decode(z / vae.config.scaling_factor)，等于把
                #   decoder 的输入放大 1/0.18215 ≈ 5.5 倍 —— 与 SD-VAE 的既定输入尺度不符，
                #   逼 decoder 去适配一个被放大的潜空间，训练出"必须喂 sample/sf"的非标准
                #   约定（后验又因 w_kl=1e-4 极宽，std≈0.88），下游 DiT 一旦用 mode 就直接灰图。
                #   标准约定 = decode(posterior.sample())，与 SD 管线 (latent=sample*sf →
                #   decode(latent/sf)=decode(sample)) 一致。
                posterior = vae.encode(x_real).latent_dist
                z = posterior.sample()
                x_recon = vae.decode(z).sample

                # 基础损失
                loss_l1 = F.l1_loss(x_recon, x_real)
                loss_kl = kl_divergence(posterior)

                # DINO 感知损失与笔法方差损失
                loss_struct, loss_style = dino_loss_fn(x_real, x_recon)

                # 综合目标 (按累积步数缩放)
                loss_total = (args.w_l1 * loss_l1 
                              + args.w_kl * loss_kl 
                              + args.w_struct * loss_struct 
                              + args.w_style * loss_style)
                loss = loss_total / args.grad_accum

            loss.backward()

            accum_l1 += loss_l1.item()
            accum_struct += loss_struct.item()
            accum_style += loss_style.item()
            accum_kl += loss_kl.item()
            accum_total += loss_total.item()

            # 当达到累积步数时执行参数更新
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                # 打印日志 (每 20 个优化器步)
                if step % 20 == 0:
                    dt = time.time() - t_start
                    sps = step / max(dt, 1e-4)
                    lr = scheduler.get_last_lr()[0]
                    avg_l1 = accum_l1 / args.grad_accum
                    avg_struct = accum_struct / args.grad_accum
                    avg_style = accum_style / args.grad_accum
                    avg_kl = accum_kl / args.grad_accum
                    avg_total = accum_total / args.grad_accum

                    msg = (f"[{datetime.datetime.now():%H:%M:%S}] "
                           f"step={step:05d}/{args.max_steps} (ep={epoch}) | "
                           f"L1: {avg_l1:.4f} | "
                           f"Struct: {avg_struct:.4f} | "
                           f"Style: {avg_style:.4f} | "
                           f"KL: {avg_kl:.4f} | "
                           f"Std: {posterior.std.mean().item():.4f} | "
                           f"Total: {avg_total:.4f} | "
                           f"LR: {lr:.2e} | "
                           f"OptSPS: {sps:.2f}")
                    print(msg, flush=True)
                    log_file.write(msg + "\n")
                    log_file.flush()

                accum_l1 = 0.0
                accum_struct = 0.0
                accum_style = 0.0
                accum_kl = 0.0
                accum_total = 0.0

                # 保存可视化对比栅格 (每 500 步)
                if step % args.vis_every == 0:
                    vae.eval()
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            val_post = vae.encode(fixed_val_batch).latent_dist
                            val_recon = vae.decode(val_post.sample()).sample
                    vis_p = os.path.join(vis_dir, f"recon_step_{step:05d}.png")
                    save_visual_grid(fixed_val_batch, val_recon, vis_p, num_show=8)
                    print(f"  [海报] 已保存对照栅格 -> {vis_p}", flush=True)
                    vae.train()

                # 保存里程碑权重 (每 2500 步)
                if step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.output, f"calli_vae_step_{step:05d}")
                    vae.save_pretrained(ckpt_dir)
                    print(f"  [权重] 里程碑权重已保存至 -> {ckpt_dir}", flush=True)

    # ★ 完训自检: 训练后必须回到标准 decode(sample) 约定 (旧版正是在此跑成 sample/sf)。
    verify_decode_convention(vae, fixed_val_batch, device, vae.config.scaling_factor)

    # 完训保存最终权重
    final_dir = os.path.join(args.output, "calli_vae_final")
    vae.save_pretrained(final_dir)
    print(f"\n✓ Calli-VAE 训练圆满完成！最终权重已固化至: {final_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
