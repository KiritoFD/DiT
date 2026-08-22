# -*- coding: utf-8 -*-
"""
train_controlnet.py — 训练 ControlNet (latent DiT + skel 条件).

流程:
  1. 加载已训练 latent 主模型 (DiT-2Cond-S/2, latent 4×32×32)
  2. 冻结主模型, 只训练 ctrl_encoder + zero_convs
  3. 数据: latent shards + 3px skel (final_skeleton_d3)
  4. 训练: 扩散 loss, cond=skel(1,256,256) 二值图, 条件 dropout→0
  5. 零注入 → 完美 warm-start

用法:
  python tools/controlnet/train_controlnet.py --config ctrl_skel.json
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import datetime
import logging
import math
import glob
import time
import re

from models import DiT_2Cond
from diffusion import create_diffusion
from tools.controlnet.controlnet_dit import ControlNetDiT

logging.basicConfig(level=logging.INFO, format='[\033[34m%(asctime)s\033[0m] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("ctrl")
sys.stdout.reconfigure(encoding="utf-8")


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def _str_to_bool(v):
    return str(v).lower() in ("true", "1", "yes", "on")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--main-ckpt", default="")
    ap.add_argument("--csv", default="5script/train_top6.csv")
    ap.add_argument("--latent-shards-dir", default="final_latents")
    ap.add_argument("--skel-root", default="final_skeleton_d3",
                    help="3px skel 目录 (final_skeleton_d3)")
    ap.add_argument("--results-dir", default="5script/results/ctrl_skel")
    ap.add_argument("--experiment-name", default="ctrl-skel")
    ap.add_argument("--model", default="DiT-2Cond-S/2")
    ap.add_argument("--num-calligraphers", type=int, default=1011)
    ap.add_argument("--num-characters", type=int, default=35130)
    ap.add_argument("--condition-fusion", default="factorized_add")
    ap.add_argument("--callig-embed-dim", type=int, default=128)
    ap.add_argument("--char-embed-dim", type=int, default=256)
    ap.add_argument("--cond-drop-all-prob", type=float, default=0.05)
    ap.add_argument("--cond-drop-one-prob", type=float, default=0.25)
    ap.add_argument("--cond-drop-struct-prob", type=float, default=0.1,
                    help="结构条件 (skel) 随机置零概率, 让模型学到无条件也能生成")
    ap.add_argument("--epochs", type=int, default=1400)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--ckpt-keep", type=int, default=3)
    ap.add_argument("--use-checkpoint", type=_str_to_bool, default=False)
    ap.add_argument("--preload", type=_str_to_bool, default=True)
    ap.add_argument("--preload-workers", type=int, default=16)
    ap.add_argument("--use-ema", type=_str_to_bool, default=True)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--ema-warmup", type=_str_to_bool, default=True)
    ap.add_argument("--skel-cond-channels", type=int, default=1,
                    help="结构条件通道数: 1=只 skel")
    args, _ = ap.parse_known_args()
    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        known = {a.dest: a for a in ap._actions if a.dest != "help"}
        for k, v in cfg.items():
            if k in known:
                setattr(args, k, v)
    return args


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    # ---- 数据: latent shards + 3px skel ----
    from latent_dataset import MCCDLatentDataset
    logger.info("[data] loading latent + skel(3px) ...")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        img_root=None, skel_root=args.skel_root,
        image_size=256, load_canny=False, load_skel=True,
        is_train=True, preload=bool(args.preload), load_image=False,
        num_preload_workers=args.preload_workers, structure_size=256)
    n = len(ds)
    logger.info(f"[data] {n} samples")

    # ---- 主模型 (latent DiT-2Cond-S/2) ----
    main_model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=384, depth=12, num_heads=6,
        num_calligraphers=args.num_calligraphers,
        num_characters=args.num_characters,
        condition_fusion=args.condition_fusion,
        callig_embed_dim=args.callig_embed_dim,
        char_embed_dim=args.char_embed_dim,
        cond_drop_all_prob=args.cond_drop_all_prob,
        cond_drop_one_prob=args.cond_drop_one_prob,
        use_checkpoint=args.use_checkpoint, learn_sigma=True)
    if args.main_ckpt and os.path.exists(args.main_ckpt):
        ck = torch.load(args.main_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck.get("delta")
        missing, unexpected = main_model.load_state_dict(sd, strict=False)
        logger.info(f"[main] loaded {args.main_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        logger.warning(f"[main] ckpt not found: {args.main_ckpt}")
    main_model.to(device)

    ctrl = ControlNetDiT(main_model, cond_in_channels=args.skel_cond_channels,
                        train_ctrl_only=True).to(device)
    trainable = [p for p in ctrl.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info(f"[ctrl] trainable params: {n_train:,}")

    # EMA on control branch only
    ema_ctrl = None
    if args.use_ema:
        ema_ctrl = copy.deepcopy(ctrl).eval()
        requires_grad(ema_ctrl, False)

    diffusion = create_diffusion(timestep_respacing="")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_steps
    warmup = args.warmup_steps

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return max(args.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * prog)))
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=lr_at)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(args.results_dir, f"{ts}-{args.experiment_name}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"results: {exp_dir}")

    # DataLoader
    from torch.utils.data import DataLoader, DistributedSampler
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0, prefetch_factor=4)

    step = 0
    t_start = time.time()
    log_steps = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            x_latent = batch['latent'].to(device)
            y_callig = batch['y_callig'].to(device)
            y_char = batch['y_char'].to(device)
            skel = batch['skeleton'].to(device).float()  # (N,1,256,256) 0/1
            # skel 条件 dropout
            if args.cond_drop_struct_prob > 0:
                drop = torch.rand(x_latent.shape[0], device=device) < args.cond_drop_struct_prob
                skel = torch.where(drop.view(-1, 1, 1, 1).expand_as(skel),
                                  torch.zeros_like(skel), skel)
            t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=device)
            model_kwargs = dict(y_callig=y_callig, y_char=y_char, cond=skel)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(ctrl, x_latent, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            if not torch.isfinite(loss):
                logger.warning(f"[nan] step {step}; skip")
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if ema_ctrl is not None:
                if step < 2000:
                    ema_decay = 0.9999 * (1 - (1 - step / 2000) ** 4)
                else:
                    ema_decay = args.ema_decay
                update_ema(ema_ctrl, ctrl, ema_decay)

            running_loss += loss.item()
            log_steps += 1
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t_start
                sps = log_steps / max(dt, 1e-6)
                logger.info(f"(step={step:07d}) loss={running_loss/log_steps:.4f} | "
                            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                            f"Steps/Sec: {sps:.2f} | "
                            f"Mem: {torch.cuda.memory_reserved()/1024**3:.2f}G")
                running_loss = 0.0
                log_steps = 0
                t_start = time.time()

            if step % args.ckpt_every == 0:
                ck = {
                    "ctrl": {k: v.detach().cpu() for k, v in ctrl.state_dict().items() if "ctrl_encoder" in k or "out_proj" in k},
                    "ema": {k: v.detach().cpu() for k, v in ema_ctrl.state_dict().items() if "ctrl_encoder" in k or "out_proj" in k} if ema_ctrl else None,
                    "train_steps": step,
                    "args": args,
                    "saved_at": datetime.datetime.now().isoformat(),
                }
                torch.save(ck, os.path.join(ckpt_dir, f"{step:07d}.pt"))
                logger.info(f"[save] {step}")
                if args.ckpt_keep > 0:
                    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
                    for p in pts[:-args.ckpt_keep]:
                        os.remove(p)

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    logger.info("Done!")


if __name__ == "__main__":
    import copy
    main()