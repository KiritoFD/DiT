"""train_unet_fm.py — Clean, dedicated training script for Flow Matching U-Net (v91).

Follows Next-Gen FM U-Net proposal:
- Preserves Flow Matching (linear interpolant x_t = (1-t)x0 + t*eps, v* = eps - x0, t * 1000)
- Multi-scale U-Net backbone with GroupNorm + SiLU + SDPA
- Multi-scale spatial STD condition + Writer/Script FiLM modulation
- Channels Last & torch.compile support
- Gradient accumulation & micro-batch support for clean scaling
- Periodic EMA updating and in-memory evaluation (eval200fix)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.eval.sampler import build_diffusion
from src.loss.flow_matching import TIME_SCALE
from src.model.build import build_unet_from_args
from src.train.ema import update_ema
from src.utils.latent_dataset import MCCDLatentDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching U-Net")
    parser.add_argument("--config", type=str, default="", help="Path to JSON config file")

    # Experiment & Paths
    parser.add_argument("--experiment-name", type=str, default="v91_unet_fm_std")
    parser.add_argument("--results-dir", type=str, default="assets/results/v91_unet_fm_std")
    parser.add_argument("--data-csv", type=str, default="exp-std/csv/train_top10_aug_sym.csv")
    parser.add_argument("--latent-shards-dir", type=str, default="exp-std/data/shards_img_aug")
    parser.add_argument("--img-root", type=str, default="data/top10_style23/imgs")
    parser.add_argument("--eval-csv", type=str, default="exp-std/csv/eval200_fixed.csv")
    parser.add_argument("--in-mem-eval-sets", type=str, default="eval200fix:exp-std/csv/eval200_fixed.csv:187")

    # STD Soft Condition
    parser.add_argument("--std-soft-cond", action="store_true", default=True)
    parser.add_argument("--std-soft-root", type=str, default="exp-std/data/std_soft_glyphs_v1")
    parser.add_argument("--std-soft-size", type=int, default=64)
    parser.add_argument("--std-soft-sigma-min", type=float, default=0.4)
    parser.add_argument("--std-soft-sigma-max", type=float, default=1.0)
    parser.add_argument("--std-soft-eval-sigma", type=float, default=0.65)

    # Model Architecture
    parser.add_argument("--model", type=str, default="FM-UNet-128")
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae-downscale", type=int, default=8)
    parser.add_argument("--unet-base-channels", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=512)
    parser.add_argument("--num-calligraphers", type=int, default=10)
    parser.add_argument("--callig-embed-dim", type=int, default=32)
    parser.add_argument("--num-script-classes", type=int, default=3)
    parser.add_argument("--script-embed-dim", type=int, default=16)
    parser.add_argument("--time-freq-dim", type=int, default=256)

    # Remaps & Tables
    parser.add_argument("--callig-remap-json", type=str, default="assets/triple_tables_best_minimal/callig_remap.json")
    parser.add_argument("--char-remap-json", type=str, default="assets/triple_tables_best_minimal/char_remap.json")
    parser.add_argument("--font-remap-json", type=str, default="assets/triple_tables_best_minimal/font_remap.json")
    parser.add_argument("--triple-table-prefix", type=str, default="assets/triple_tables_best_minimal/")

    # Flow Matching & Training Dynamics
    parser.add_argument("--diffusion-type", type=str, default="flow")
    parser.add_argument("--flow-sampler", type=str, default="heun")
    parser.add_argument("--t-sampler", type=str, default="logit_normal")
    parser.add_argument("--t-mean", type=float, default=0.0)
    parser.add_argument("--t-std", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=50)

    # Optimization
    parser.add_argument("--batch-size", type=int, default=64, help="Micro-batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # EMA & Checkpointing
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-interval", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", type=str, default="")

    # Performance & Infra
    parser.add_argument("--channels-last", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Merge config file if provided
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            attr = k.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, v)

    return args


def setup_logger(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("train_unet_fm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine_decay


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Output directory
    exp_dir = os.path.join(args.results_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{args.experiment_name}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = setup_logger(exp_dir)
    logger.info(f"=== Starting Flow Matching U-Net Training ===")
    logger.info(f"Experiment Name: {args.experiment_name}")
    logger.info(f"Device: {device} | Channels Last: {args.channels_last} | Compile: {args.compile}")
    logger.info(f"Micro-batch: {args.batch_size} | Grad Accum: {args.grad_accum_steps} | Effective Batch: {args.batch_size * args.grad_accum_steps}")

    # Save effective config
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 1. Build Model
    model = build_unet_from_args(args, device=device)
    model_ema = copy.deepcopy(model)
    model_ema.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model Built: {args.model} | Total Parameters: {total_params:.2f} M")

    # Load remaps & pretrained tables if available
    char_remaps = {}
    for remap_key in ["callig", "char", "font"]:
        p = getattr(args, f"{remap_key}_remap_json", None)
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                char_remaps[remap_key] = json.load(f)

    # Load pretrained tables into cond_module
    prefix = getattr(args, "triple_table_prefix", "")
    if prefix:
        callig_path = os.path.join(prefix, "E_callig.npy")
        font_path = os.path.join(prefix, "E_script.npy")
        model.cond_module.load_pretrained_tables(
            callig_path if os.path.isfile(callig_path) else None,
            font_path if os.path.isfile(font_path) else None,
        )
        model_ema.cond_module.load_pretrained_tables(
            callig_path if os.path.isfile(callig_path) else None,
            font_path if os.path.isfile(font_path) else None,
        )
        logger.info("[condition] Pretrained embedding tables loaded into WriterScriptCondition")

    # Memory format & compilation
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model_ema = model_ema.to(memory_format=torch.channels_last)
        logger.info("[infra] Set memory format to torch.channels_last")

    if args.compile and device.type == "cuda":
        logger.info(f"[infra] Compiling model with torch.compile(mode='{args.compile_mode}')")
        model = torch.compile(model, mode=args.compile_mode)

    # 2. Flow Matching Diffusion
    diffusion = build_diffusion(
        args.eval_steps,
        diffusion_type="flow",
        flow_kwargs={
            "sampler": args.flow_sampler,
            "shift": args.shift,
            "t_sampler": args.t_sampler,
            "t_mean": args.t_mean,
            "t_std": args.t_std,
        },
    )

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 4. Dataset & DataLoader
    dataset = MCCDLatentDataset(
        csv_file=args.data_csv,
        latent_shards_dir=args.latent_shards_dir,
        img_root=args.img_root,
        image_size=args.image_size,
        is_train=True,
        preload=True,
        load_image=False,
        char_remap=char_remaps.get("char"),
        callig_remap=char_remaps.get("callig"),
        font_remap=char_remaps.get("font"),
        std_soft_root=args.std_soft_root,
        std_soft_sigma_min=args.std_soft_sigma_min,
        std_soft_sigma_max=args.std_soft_sigma_max,
        std_soft_eval_sigma=args.std_soft_eval_sigma,
    )
    logger.info(f"Dataset preloaded: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Resume if requested
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        model_ema.load_state_dict(ckpt["ema"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("train_steps", 0)
        logger.info(f"Resumed from step {start_step}")

    # Training Loop
    model.train()
    step = start_step
    data_iter = iter(loader)
    running_loss = 0.0
    start_time = time.time()

    logger.info("Entering training loop...")
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        current_lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr_ratio)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for micro_idx in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x0 = batch["latent"].to(device, non_blocking=True).float()
            y_c = batch["y_callig"].to(device, non_blocking=True).long()
            y_s = batch["y_script"].to(device, non_blocking=True).long()
            g = batch["std_soft"].to(device, non_blocking=True).float()

            if args.channels_last and device.type == "cuda":
                x0 = x0.to(memory_format=torch.channels_last)

            # Sample Flow Matching time
            t = diffusion.sample_t(x0.shape[0], device=device).float()
            noise = torch.randn_like(x0)
            x_t = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
            v_target = noise - x0

            # Forward
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred_v = model(
                    x_t,
                    t * TIME_SCALE,
                    y_callig=y_c,
                    g=g,
                    y_script=y_s,
                )
                loss = F.mse_loss(pred_v, v_target)
                if args.grad_accum_steps > 1:
                    loss = loss / args.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping & step
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Update EMA
        if step % args.ema_interval == 0:
            update_ema(model_ema, model, decay=args.ema_decay)

        running_loss += accum_loss
        step += 1

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = args.log_every / max(1e-5, elapsed)
            avg_loss = running_loss / args.log_every
            running_loss = 0.0
            start_time = time.time()

            vram_info = ""
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                res = torch.cuda.memory_reserved() / (1024 ** 3)
                vram_info = f" | VRAM: {alloc:.2f}G/{res:.2f}G"

            logger.info(
                f"(step={step:07d}) FM-Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                f"Steps/Sec: {steps_per_sec:.2f}{vram_info}"
            )

        # Checkpointing & In-Memory Evaluation
        if step % args.ckpt_every == 0 or step == args.max_steps:
            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"{step:07d}.pt")
            raw_model = getattr(model, "_orig_mod", model)
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "ema": model_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_steps": step,
                    "args": vars(args),
                },
                ckpt_path,
            )
            # Touch .done file
            with open(f"{ckpt_path}.done", "w", encoding="utf-8") as f:
                f.write(f"step={step}\n")
            logger.info(f"[ckpt] Saved checkpoint at step {step}: {ckpt_path}")

            # Run in-memory evaluation on eval200fix
            try:
                from src.eval.in_mem_eval import run_in_mem_eval
                eval_out_dir = os.path.join(exp_dir, "eval_auto")
                os.makedirs(eval_out_dir, exist_ok=True)
                metrics = run_in_mem_eval(
                    model_ema, args, step, device,
                    eval_out_dir, logger=logger.info,
                )
                logger.info(f"[eval-auto] Step {step} Metrics: {metrics}")
            except Exception as e:
                logger.warning(f"[eval-auto] In-mem eval encountered exception: {e}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
