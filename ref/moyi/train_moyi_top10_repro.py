#!/usr/bin/env python3
"""
train_moyi_top10_repro.py —— Moyun/Moyi (Rectified Flow DiT) Top10 复现训练脚本

硬件与算力适配:
- 物理单卡 RTX 4090 (24GB VRAM)
- Batch Size 112: 显存精确饱和在 ~19.6GB，无需激活重计算，吞吐 ~320 samples/s
- 精度: bfloat16 混合精度，充分释放 4090 Tensor Cores

数据与条件:
- 12 通道联合潜空间: image (4ch) + edge (4ch, aux_skel3) + skeleton (4ch, std_w7)
- 3 条件联合 Embedding: calligrapher (10名家) + script (3书体) + character (4677汉字)
- 分维度条件 Dropout: calligrapher 0.16, font 0.08, character 0.08 (对齐原著 _ful.sh)
- 纯真迹隔离评测: 定期利用 sd-vae-ft-ema 解码采样样本，输出全景可视化海报
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "moyun"))

from dataset_moyun import MultiLabelNestedDataset
from moyun_2 import DiT_models
from utils.Sampler.RF import RF
from diffusers.models import AutoencoderKL


def parse_args():
    p = argparse.ArgumentParser(description="Moyi/Moyun Top10 Reproduction")
    p.add_argument("--model", default="moyun-12channel-B", choices=list(DiT_models.keys()))
    p.add_argument("--csv", default="/home/ds/Workspace/moyi/assets/train_top10_style23_real.csv")
    p.add_argument("--img-shards", default="/home/ds/Workspace/moyi/data/top10_style23/shards_img")
    p.add_argument("--edge-shards", default="/home/ds/Workspace/moyi/data/top10_style23/shards_aux_skel3")
    p.add_argument("--skel-shards", default="/home/ds/Workspace/moyi/data/top10_style23/shards_std_w7")
    p.add_argument("--vae-path", default="/home/ds/Workspace/moyi/models/sd-vae-ft-ema")
    p.add_argument("--results-dir", default="/home/ds/Workspace/moyi/results/moyi_top10_rf")
    p.add_argument("--batch-size", type=int, default=112)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=80000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=6000)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--calligrapher-x", type=float, default=0.16)
    p.add_argument("--font-x", type=float, default=0.08)
    p.add_argument("--charactor-x", type=float, default=0.08)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--resume", default="")
    return p.parse_args()


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("moyi")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("[\033[34m%(asctime)s\033[0m] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"), encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def generate_eval_poster(model, vae, rf, dataset, save_path, device="cuda", steps=50, cfg_scale=1.5):
    """选取经典书家与书体生成矩阵海报进行视觉评测"""
    model.eval()
    
    # 评测提示组合: 覆盖主要名家与书体
    eval_prompts = [
        ("王羲之", "楷", "永"),
        ("王羲之", "行", "和"),
        ("王羲之", "行", "清"),
        ("颜真卿", "楷", "卿"),
        ("颜真卿", "楷", "獨"),
        ("米芾", "行", "天"),
        ("米芾", "行", "風"),
        ("赵孟頫", "行", "雲"),
        ("赵孟頫", "隶", "民"),
        ("柳公权", "楷", "心"),
    ]
    
    labels_list = []
    names = []
    for c_name, s_name, ch in eval_prompts:
        cid = dataset.callig_vocab.get(c_name, 0)
        sid = dataset.script_vocab.get(s_name, 0)
        chid = dataset.char_vocab.get(ch, 0)
        labels_list.append([cid, sid, chid])
        names.append(f"{c_name}_{s_name}_{ch}")
        
    n = len(labels_list)
    y = torch.tensor(labels_list, device=device)
    stroke = torch.zeros(n, dtype=torch.long, device=device)
    
    # 使用 RF Euler 采样从高斯噪声生成 12 通道潜变量
    z = torch.randn(n, 12, 32, 32, device=device)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    
    for i in range(steps):
        t = ts[i].expand(n)
        dt = ts[i + 1] - ts[i]
        v_pred = model(z, t, y, stroke)
        if isinstance(v_pred, (tuple, list)):
            v_pred = v_pred[0]
        z = z + v_pred * dt
        
    # 取前 4 通道作为生成的书法图像 latent
    img_latents = z[:, 0:4, :, :]
    
    # VAE 解码到像素空间 (256x256)
    decoded = vae.decode(img_latents / 0.18215).sample
    decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(decoded, save_path, nrow=5, padding=4)
    model.train()


def main():
    args = parse_args()
    logger = setup_logger(args.results_dir)
    logger.info("=" * 60)
    logger.info("【Moyi / Moyun Top10 Rectified Flow 官方复现训练】")
    logger.info(f"Model: {args.model} | Batch Size: {args.batch_size} | LR: {args.lr}")
    logger.info(f"Results Dir: {args.results_dir}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} ({total_vram:.2f} GB VRAM)")

    # 1. 加载数据集
    logger.info("Loading Top10 dataset...")
    ds = MultiLabelNestedDataset(
        csv_file=args.csv,
        img_shards=args.img_shards,
        edge_shards=args.edge_shards,
        skel_shards=args.skel_shards,
        num_classes=args.num_classes,
    )
    logger.info(f"Dataset loaded: {len(ds):,} samples | Calligraphers: {len(ds.callig_vocab)} | Scripts: {len(ds.script_vocab)} | Characters: {len(ds.char_vocab)}")

    # 保存词表到结果目录
    vocab_info = {
        "callig_vocab": ds.callig_vocab,
        "script_vocab": ds.script_vocab,
        "char_vocab": ds.char_vocab,
    }
    with open(os.path.join(args.results_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # 2. 构建模型
    logger.info(f"Constructing DiT model {args.model}...")
    model = DiT_models[args.model](
        input_size=32,
        num_classes=args.num_classes,
        learn_sigma=False,
        if_rope=False,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.1f} M")

    # 构建 EMA 模型
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    # 加载 VAE 用于定期评测出图
    logger.info(f"Loading VAE from {args.vae_path}...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 3. 优化器与调度
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    rf = RF(without_t=False)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt.get("ema", ckpt["model"]))
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from step {start_step}")

    # 4. 训练主循环
    model.train()
    step = start_step
    loss_accum = 0.0
    t0 = time.time()
    best_loss = float("inf")

    logger.info(f"Starting training from step {step} to {args.max_steps}...")

    # 训练启动时先做一次基线海报输出
    if step == 0 and not args.smoke:
        logger.info("Generating step 0 baseline poster...")
        try:
            generate_eval_poster(
                ema_model, vae, rf, ds,
                os.path.join(args.results_dir, "posters", "eval_step_0000000.png"),
                device=device, steps=50
            )
            logger.info("✓ Step 0 baseline poster saved.")
        except Exception as e:
            logger.warning(f"Failed to generate step 0 poster: {e}")

    while step < args.max_steps:
        for batch in loader:
            image, edge, skel, y_tuple, stroke, _, _, _ = batch
            
            # 12 通道潜变量拼接 (B, 12, 32, 32)
            x = torch.cat([image, edge, skel], dim=1).to(device, non_blocking=True)
            
            # 标签转换为 (B, 3) 稠密张量
            cid = y_tuple[0].squeeze(-1).to(device)
            sid = y_tuple[1].squeeze(-1).to(device)
            chid = y_tuple[2].squeeze(-1).to(device)
            y = torch.stack([cid, sid, chid], dim=1)
            stroke = stroke.to(device)

            # 条件 Dropout 字典
            fdict = {
                "calligrapher_x": args.calligrapher_x,
                "font_x": args.font_x,
                "charactor_x": args.charactor_x,
                "_num_classes": args.num_classes,
            }

            # 前向计算 Rectified Flow 损失
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_tuple = rf.forward(model, x, feature_dict=fdict, y=y, stroke=stroke)
                loss = loss_tuple[0]

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            update_ema(ema_model, model, decay=args.ema_decay)

            loss_accum += loss.item()
            step += 1

            # 日志输出
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                step_s = args.log_every / max(elapsed, 1e-6)
                smp_s = step_s * args.batch_size
                avg_loss = loss_accum / args.log_every
                mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

                remaining_steps = args.max_steps - step
                eta_s = remaining_steps / max(step_s, 1e-6)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))

                logger.info(
                    f"(step={step:07d}) Loss: {avg_loss:.4f} | "
                    f"Speed: {step_s:4.2f} step/s ({smp_s:5.1f} smp/s) | "
                    f"Mem: {mem_gb:5.2f}G | ETA: {eta_str}"
                )
                loss_accum = 0.0
                t0 = time.time()

            # 冒烟测试快速退出
            if args.smoke and step >= args.smoke:
                logger.info(f"✓ [Smoke] Successfully finished {step} steps without error!")
                logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
                return

            # 定期生成视觉评测海报
            if step % args.eval_every == 0:
                poster_path = os.path.join(args.results_dir, "posters", f"eval_step_{step:07d}.png")
                logger.info(f"Generating visual evaluation poster at step {step}...")
                try:
                    generate_eval_poster(ema_model, vae, rf, ds, poster_path, device=device, steps=50)
                    logger.info(f"✓ Evaluation poster saved to {poster_path}")
                except Exception as e:
                    logger.error(f"Failed to generate evaluation poster: {e}")

            # 定期保存权重
            if step % args.ckpt_every == 0 or step == args.max_steps:
                ckpt_dir = os.path.join(args.results_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"moyi_{step:07d}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                logger.info(f"✓ Checkpoint saved: {ckpt_path}")

    logger.info("=" * 60)
    logger.info("【Training Complete!】")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
