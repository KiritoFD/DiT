# -*- coding: utf-8 -*-
"""
train_vae.py — 在 MCCD 数据上微调或从头训练 VAE.

支持三种模式:
  1. finetune-decoder: 冻结 encoder, 只训练 decoder (方案 B)
  2. finetune-full: 全量微调 (encoder+decoder)
  3. from-scratch: 从 kl-f4 架构出发, 改为 1ch 黑白, 从头训 (方案 C)

Loss: L1 recon + KL(0.5) + perceptual(VGG, 可选)

用法:
  # 微调 decoder
  python tools/vae/train_vae.py --mode finetune-decoder --vae pretrained_models/sd-vae-ft-ema --csv 5script/train_top30.csv --epochs 20

  # 从头训黑白 VAE (f4, 1ch)
  python tools/vae/train_vae.py --mode from-scratch --arch kl-f4 --grayscale --csv 5script/train_top30.csv --epochs 100
"""
import os
import sys
import argparse
import json
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class ImageDataset(Dataset):
    def __init__(self, csv_path, img_root="final_images", size=256, grayscale=False):
        import csv
        self.img_root = img_root
        self.size = size
        self.grayscale = grayscale
        self.ids = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("img_id") or row.get("id")
                if img_id:
                    path = os.path.join(img_root, f"{img_id}.png")
                    if os.path.exists(path):
                        self.ids.append(img_id)
        print(f"[data] {len(self.ids)} images from {csv_path}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.img_root, f"{img_id}.png")
        img = Image.open(path)
        if self.grayscale:
            img = img.convert("L")
            t = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),  # [0, 1]
            ])(img)
            t = t * 2 - 1  # [-1, 1]
            t = t.repeat(3, 1, 1)  # → 3ch for VAE (encoder conv_in=3ch)
        else:
            img = img.convert("RGB")
            t = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
            ])(img)
            t = t * 2 - 1
        return {"image": t, "id": img_id}


def kl_loss(mu, logvar):
    """KL divergence: D_KL(q(z|x) || N(0,1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]).mean()


def perceptual_loss(x, y):
    """Simple perceptual loss using VGG16 features."""
    try:
        from torchvision.models import vgg16, VGG16_Weights
        if not hasattr(perceptual_loss, "_vgg"):
            perceptual_loss._vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
            for p in perceptual_loss._vgg.parameters():
                p.requires_grad = False
        vgg = perceptual_loss._vgg.to(x.device)
        # VGG expects normalized input
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_n = (x + 1) / 2
        y_n = (y + 1) / 2
        x_n = (x_n - mean) / std
        y_n = (y_n - mean) / std
        return F.mse_loss(vgg(x_n), vgg(y_n))
    except Exception:
        return torch.tensor(0.0, device=x.device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}, mode={args.mode}")

    # Load VAE
    from diffusers import AutoencoderKL
    if args.mode == "from-scratch":
        # Build from config (kl-f4 architecture, optionally 1ch)
        in_ch = 1 if args.grayscale else 3
        config = {
            "sample_size": 256,
            "in_channels": in_ch,
            "out_channels": in_ch,
            "latent_channels": args.latent_channels,
            "block_out_channels": [128, 256, 512],
            "layers_per_block": 2,
            "down_block_types": ["DownEncoderBlock2D"] * 3,
            "up_block_types": ["UpDecoderBlock2D"] * 3,
            "act_fn": "silu",
            "norm_num_groups": 32,
            "scaling_factor": 0.18215,
        }
        vae = AutoencoderKL(**config)
        print(f"[model] from-scratch: {in_ch}ch, latent={args.latent_channels}ch, f4")
    else:
        vae = AutoencoderKL.from_pretrained(args.vae)
        if args.grayscale:
            # Modify conv_in to 1ch: keep mean of 3 input channels
            with torch.no_grad():
                w = vae.encoder.conv_in.weight.data.mean(dim=1, keepdim=True)
                b = vae.encoder.conv_in.bias.data
                vae.encoder.conv_in = nn.Conv2d(1, w.shape[0], 3, 1, 1)
                vae.encoder.conv_in.weight.data = w
                vae.encoder.conv_in.bias.data = b
                w2 = vae.decoder.conv_out.weight.data.mean(dim=0, keepdim=True)
                b2 = vae.decoder.conv_out.bias.data
                vae.decoder.conv_out = nn.Conv2d(w2.shape[1], 1, 3, 1, 1)
                vae.decoder.conv_out.weight.data = w2
                vae.decoder.conv_out.bias.data = b2

    vae = vae.to(device)

    if args.mode == "finetune-decoder":
        for p in vae.encoder.parameters():
            p.requires_grad = False
        for p in vae.quant_conv.parameters():
            p.requires_grad = False
        for p in vae.post_quant_conv.parameters():
            p.requires_grad = False
        trainable = [p for p in vae.decoder.parameters() if p.requires_grad]
    else:
        trainable = [p for p in vae.parameters() if p.requires_grad]

    n_params = sum(p.numel() for p in trainable)
    print(f"[model] trainable params: {n_params/1e6:.1f}M")

    # Data
    ds = ImageDataset(args.csv, args.img_root, args.size, args.grayscale)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=True)

    # Optimizer
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))

    # Training loop
    step = 0
    os.makedirs(args.output, exist_ok=True)
    log_file = open(os.path.join(args.output, "train_vae.log"), "w")

    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["image"].to(device)
            # For grayscale, input is already repeated to 3ch in dataset
            # But if VAE is 1ch, we need to squeeze
            in_ch_vae = vae.config.in_channels
            if in_ch_vae == 1 and x.shape[1] == 3:
                x = x[:, :1]  # take first channel

            # Encode
            posterior = vae.encode(x).latent_dist
            z = posterior.sample()
            kl = kl_loss(posterior.mean, posterior.logvar)

            # Decode
            dec = vae.decode(z / vae.config.scaling_factor).sample
            l1 = F.l1_loss(dec, x)
            mse = F.mse_loss(dec, x)
            perc = perceptual_loss(dec, x) if args.w_perceptual > 0 else torch.tensor(0.0, device=device)

            loss = l1 + args.w_kl * kl + args.w_perceptual * perc

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                msg = (f"[{datetime.datetime.now():%H:%M:%S}] epoch={epoch} step={step} "
                       f"L1={l1.item():.4f} KL={kl.item():.4f} MSE={mse.item():.6f} "
                       f"perc={perc.item():.4f} total={loss.item():.4f}")
                print(msg, flush=True)
                log_file.write(msg + "\n")
                log_file.flush()

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output, f"vae_epoch_{epoch+1:04d}")
            vae.save_pretrained(ckpt_path)
            print(f"[save] {ckpt_path}", flush=True)

    # Final save
    vae.save_pretrained(os.path.join(args.output, "vae_final"))
    print(f"[done] saved to {args.output}/vae_final")
    log_file.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finetune-decoder", "finetune-full", "from-scratch"],
                    default="finetune-decoder")
    ap.add_argument("--vae", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--csv", default="5script/train_top30.csv")
    ap.add_argument("--img-root", default="final_images")
    ap.add_argument("--output", default="5script/results/vae")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--w-kl", type=float, default=0.5)
    ap.add_argument("--w-perceptual", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()
    train(args)
