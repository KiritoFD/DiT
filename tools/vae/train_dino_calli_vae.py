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

    sd = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            # 兼容 transformers 5.x 命名 (q_proj/k_proj/v_proj/o_proj) 与旧命名
            new_k = k
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
    parser.add_argument("--batch-size", type=int, default=64,
                        help="训练批次大小 (48G 显存推荐 64)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
    os.makedirs(args.output, exist_ok=True)
    vis_dir = os.path.join(args.output, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("【启动纯 DINO-REPA 监督 Calli-VAE 全量微调】")
    print(f"  基线 VAE: {args.vae_base}")
    print(f"  DINO 裁判: {args.dino_ckpt}")
    print(f"  训练数据: {args.csv}")
    print(f"  Batch: {args.batch_size} | 学习率: {args.lr} | 目标步数: {args.max_steps}")
    print(f"  损失配方: {args.w_l1}*L1 + {args.w_kl}*KL + {args.w_struct}*DINO_Struct + {args.w_style}*DINO_Style")
    print(f"  输出目录: {args.output}")
    print("=" * 80)

    # 1. 载入 DINOv2 裁判模型
    dino_raw = load_dino_model(args.dino_ckpt, device)
    dino_loss_fn = DINOPerceptualLoss(dino_raw, device)

    # 2. 载入基线 VAE 并解冻全量参数 (Encoder + Decoder)
    print(f"[vae] 从 {args.vae_base} 载入 AutoencoderKL ...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_base).to(device)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    # 固定的 8 张验证图 (固定观察同一批字的细节演化)
    fixed_val_batch = next(iter(dataloader))[:8].to(device)

    step = 0
    epoch = 0
    log_path = os.path.join(args.output, "train_dino_vae.log")
    log_file = open(log_path, "a", encoding="utf-8")

    t_start = time.time()
    print("\n>>> 开始全速迭代训练 ...", flush=True)

    while step < args.max_steps:
        epoch += 1
        for x_real in dataloader:
            if step >= args.max_steps:
                break
            step += 1
            x_real = x_real.to(device)

            # 前向编解码与重参数化采样
            with torch.autocast("cuda", dtype=torch.bfloat16):
                posterior = vae.encode(x_real).latent_dist
                z = posterior.sample()
                x_recon = vae.decode(z / vae.config.scaling_factor).sample

                # 基础损失
                loss_l1 = F.l1_loss(x_recon, x_real)
                loss_kl = kl_divergence(posterior)

                # DINO 感知损失与笔法方差损失
                loss_struct, loss_style = dino_loss_fn(x_real, x_recon)

                # 综合目标
                loss = (args.w_l1 * loss_l1 
                        + args.w_kl * loss_kl 
                        + args.w_struct * loss_struct 
                        + args.w_style * loss_style)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # 打印日志 (每 20 步)
            if step % 20 == 0:
                dt = time.time() - t_start
                sps = step / max(dt, 1e-4)
                lr = scheduler.get_last_lr()[0]
                msg = (f"[{datetime.datetime.now():%H:%M:%S}] "
                       f"step={step:05d}/{args.max_steps} (ep={epoch}) | "
                       f"L1: {loss_l1.item():.4f} | "
                       f"Struct: {loss_struct.item():.4f} | "
                       f"Style: {loss_style.item():.4f} | "
                       f"KL: {loss_kl.item():.4f} | "
                       f"Total: {loss.item():.4f} | "
                       f"LR: {lr:.2e} | "
                       f"SPS: {sps:.2f}")
                print(msg, flush=True)
                log_file.write(msg + "\n")
                log_file.flush()

            # 保存可视化对比栅格 (每 500 步)
            if step % args.vis_every == 0:
                vae.eval()
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        val_post = vae.encode(fixed_val_batch).latent_dist
                        val_recon = vae.decode(val_post.sample() / vae.config.scaling_factor).sample
                vis_p = os.path.join(vis_dir, f"recon_step_{step:05d}.png")
                save_visual_grid(fixed_val_batch, val_recon, vis_p, num_show=8)
                print(f"  [海报] 已保存对照栅格 -> {vis_p}", flush=True)
                vae.train()

            # 保存里程碑权重 (每 2500 步)
            if step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output, f"calli_vae_step_{step:05d}")
                vae.save_pretrained(ckpt_dir)
                print(f"  [权重] 里程碑权重已保存至 -> {ckpt_dir}", flush=True)

    # 完训保存最终权重
    final_dir = os.path.join(args.output, "calli_vae_final")
    vae.save_pretrained(final_dir)
    print(f"\n✓ Calli-VAE 训练圆满完成！最终权重已固化至: {final_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
