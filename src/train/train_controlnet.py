# -*- coding: utf-8 -*-
"""
train_controlnet.py — 训练 ControlNet (latent DiT + 3px skel 条件).

两种模式:
  A) warm-start (默认, train_ctrl_only=True): 冻结已训练主模型, 只训练 ctrl_encoder
  B) from-scratch (train_ctrl_only=False): 主模型+ctrl_encoder 一起从零训练

流程:
  1. 构建 DiT_2Cond-S/2 + ControlNetDiT 包装
  2. warm-start: 加载已训练主模型 ckpt 并冻结; from-scratch: 可选加载 pretrained body
  3. 数据: latent shards + 3px skel (final_skeleton_d3)
  4. 训练: 扩散 loss (eps-MSE), cond=skel(1,256,256) 二值图
  5. 零注入 → 完美 warm-start (初始 ctrl=0, 主模型行为不变)
  6. 结构条件 dropout (10%): 让模型学到无 skel 也能生成 (CFG 友好)

INFRA 设计:
  - from-scratch: 主模型也训练 → forward 建完整图 (主模型 33M + ctrl 33.8M)
  - warm-start: 主模型冻结, forward 不建训练图 → 只有 ctrl_encoder 建图
  - 每步 del 所有中间张量 + zero_grad → 无 graph 残留
  - 不加载 VAE (latent mode) → 省 ~500MB

用法:
  python -m src.train.train_controlnet --config src/train/configs/ctrl_skel_s18_flow.json
  (旧路径 tools/controlnet/train_controlnet.py 已迁移; 配置移到 src/train/configs/)
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import sys
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _r not in sys.path:
    sys.path.insert(0, _r)
import copy
import glob
import time
import math
import re
import logging
import datetime
import argparse
import json

import torch
import torch.nn as nn
import numpy as np

from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow
from src.utils import MCCDLatentDataset
from src.model.controlnet import ControlNetDiT, load_main_model
from src.eval.in_process_ctrl_eval import prepare_ctrl_eval_cache, run_ctrl_pair_eval

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format='[\033[34m%(asctime)s\033[0m] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("ctrl")


def _str_to_bool(v):
    return str(v).lower() in ("true", "1", "yes", "on")


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def update_ema(ema_model, model, decay):
    """EMA 只更新 trainable 参数 (requires_grad=True). frozen 参数直接 copy (保持一致)."""
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            if p.requires_grad:
                ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
            else:
                ep.data.copy_(p.data)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.data.copy_(b.data)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pretrained", default="",
                    help="from-scratch: pretrained body ckpt (e.g. DiT-XL-2-256x256.pt)")
    ap.add_argument("--main-ckpt", default="",
                    help="warm-start: 已训练主模型 ckpt (train_ctrl_only=true 时用)")
    ap.add_argument("--train-ctrl-only", type=_str_to_bool, default=True,
                    help="True=warm-start(冻结主模型), False=from-scratch(主模型也训练)")
    ap.add_argument("--csv", default="5script/train_top6.csv")
    ap.add_argument("--latent-shards-dir", default="final_latents")
    ap.add_argument("--skel-root", default="final_skeleton_d3")
    ap.add_argument("--results-dir", default="5script/results/ctrl_skel")
    ap.add_argument("--experiment-name", default="ctrl-skel-3px")
    ap.add_argument("--model", default="DiT-2Cond-S/2")
    ap.add_argument("--num-calligraphers", type=int, default=1011)
    ap.add_argument("--num-characters", type=int, default=35130)
    ap.add_argument("--condition-fusion", default="factorized_add")
    ap.add_argument("--callig-embed-dim", type=int, default=128)
    ap.add_argument("--char-embed-dim", type=int, default=256)
    ap.add_argument("--char-proj-mode", default="full",
                    help="char_proj mode: 'full' or 'ln_only' (DINO 384 passthrough)")
    ap.add_argument("--freeze-char-table", type=_str_to_bool, default=False,
                    help="freeze y_char_embedder table (DINO init)")
    ap.add_argument("--diffusion-type", default="ddpm", choices=["ddpm", "flow"],
                    help="main model diffusion type: ddpm (eps/DDIM) or flow (velocity/Euler)")
    ap.add_argument("--cond-drop-all-prob", type=float, default=0.05)
    ap.add_argument("--cond-drop-one-prob", type=float, default=0.25)
    ap.add_argument("--cond-drop-struct-prob", type=float, default=0.1,
                    help="skel 随机置零概率 (训练时), 让模型学到无 skel 也能生成")
    ap.add_argument("--skel-cond-channels", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1400)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--ckpt-keep", type=int, default=3)
    ap.add_argument("--use-ema", type=_str_to_bool, default=True)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--preload", type=_str_to_bool, default=True)
    ap.add_argument("--preload-workers", type=int, default=16)
    ap.add_argument("--use-checkpoint", type=_str_to_bool, default=False,
                    help="主模型 gradient checkpointing (省显存, 略慢)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="",
                    help="resume from ckpt path (loads model+ctrl+optimizer+step)")
    # in-process GPU eval (base vs GT-skel) + CPU metrics daemon
    ap.add_argument("--gpu-eval-csv", default="",
                    help="eval csv for in-process GPU ctrl eval; empty = disabled")
    ap.add_argument("--gpu-eval-every", type=int, default=2500)
    ap.add_argument("--gpu-eval-n", type=int, default=100)
    ap.add_argument("--gpu-eval-steps", type=int, default=50)
    ap.add_argument("--gpu-eval-cfg", type=float, default=1.7, help="flow ctrl 推理最佳 CFG ~1.7")
    ap.add_argument("--gpu-eval-img-root", default="final_imgs_256")
    ap.add_argument("--gpu-eval-skel-root", default="final_skeleton_d3")
    ap.add_argument("--gpu-eval-dit-batch", type=int, default=16)
    ap.add_argument("--gpu-eval-vae-batch", type=int, default=16)
    args, _ = ap.parse_known_args()
    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        known = {a.dest: a for a in ap._actions if a.dest != "help"}
        for k, v in cfg.items():
            if k in known:
                setattr(args, k, v)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    # ---- 数据: latent shards + 3px skel ----
    logger.info("[data] loading latent + skel(3px) ...")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        img_root=None, skel_root=args.skel_root,
        image_size=256, load_canny=False, load_skel=True,
        is_train=True, preload=bool(args.preload), load_image=False,
        num_preload_workers=args.preload_workers, structure_size=256)
    n = len(ds)
    logger.info(f"[data] {n} samples, skel from {args.skel_root}")

    # ---- 模型构建 ----
    if args.train_ctrl_only:
        # warm-start: 加载已训练主模型, 冻结
        logger.info("[model] warm-start: loading main model + freezing ...")
        main_model = load_main_model(
            model_name=args.model, ckpt_path=args.main_ckpt if args.main_ckpt and os.path.exists(args.main_ckpt) else None,
            device=device, num_calligraphers=args.num_calligraphers, num_characters=args.num_characters,
            condition_fusion=args.condition_fusion, callig_embed_dim=args.callig_embed_dim,
            char_embed_dim=args.char_embed_dim, char_proj_mode=args.char_proj_mode,
            freeze_char_table=args.freeze_char_table,
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob, use_checkpoint=args.use_checkpoint)
        main_model.eval()
        ctrl = ControlNetDiT(main_model, cond_in_channels=args.skel_cond_channels,
                            train_ctrl_only=True).to(device)
    else:
        # from-scratch: 构建新主模型, 可选加载 pretrained body
        logger.info("[model] from-scratch: building new main model ...")
        latent_size = 32  # DiT-S/2 latent
        main_model = DiT_2Cond_models[args.model](
            num_calligraphers=args.num_calligraphers, num_characters=args.num_characters,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim, char_embed_dim=args.char_embed_dim,
            cond_drop_all_prob=args.cond_drop_all_prob, cond_drop_one_prob=args.cond_drop_one_prob,
            use_checkpoint=args.use_checkpoint, learn_sigma=True)
        if args.pretrained and os.path.exists(args.pretrained):
            logger.info(f"[model] loading pretrained body from {args.pretrained}")
            sd = torch.load(args.pretrained, map_location="cpu", weights_only=False)
            if "model" in sd:
                sd = sd["model"]
            # Filter out condition heads (don't match 2Cond)
            sd = {k: v for k, v in sd.items()
                  if not k.startswith(("y_embedder", "y_callig", "y_script",
                                       "y_char", "cond_fusion"))}
            missing, unexpected = main_model.load_state_dict(sd, strict=False)
            logger.info(f"[model] pretrained body loaded (missing={len(missing)}, "
                        f"unexpected={len(unexpected)})")
        main_model.to(device)
        ctrl = ControlNetDiT(main_model, cond_in_channels=args.skel_cond_channels,
                            train_ctrl_only=False).to(device)

    trainable = [p for p in ctrl.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in ctrl.parameters() if not p.requires_grad)
    logger.info(f"[ctrl] trainable params: {n_train:,} | frozen: {n_frozen:,}")

    # EMA on trainable params (ctrl_encoder + optionally main_model)
    ema_ctrl = None
    if args.use_ema:
        ema_ctrl = copy.deepcopy(ctrl).eval()
        requires_grad(ema_ctrl, False)

    diffusion = create_diffusion_or_flow(
        timestep_respacing="", diffusion_type=args.diffusion_type)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return max(args.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * prog)))
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---- Resume from checkpoint ----
    resume_step = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"[resume] loading {args.resume}")
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Load main model weights
        if ck.get("model"):
            main_keys = {k: v for k, v in ck["model"].items() if k.startswith("main.")}
            m_miss, m_unexp = ctrl.load_state_dict(main_keys, strict=False)
            logger.info(f"[resume] main: {len(main_keys)} keys (missing={len(m_miss)}, unexpected={len(m_unexp)})")
        if ck.get("ema_model"):
            ema_main_keys = {k: v for k, v in ck["ema_model"].items() if k.startswith("main.")}
            ctrl.load_state_dict(ema_main_keys, strict=False)
            logger.info(f"[resume] ema_model: {len(ema_main_keys)} keys")
        # Load ctrl_encoder weights (prefer ema)
        ctrl_src = ck.get("ema") or ck.get("ctrl")
        if ctrl_src:
            ctrl_keys = {k: v for k, v in ctrl_src.items() if k.startswith("ctrl_encoder")}
            c_miss, c_unexp = ctrl.load_state_dict(ctrl_keys, strict=False)
            logger.info(f"[resume] ctrl_encoder: {len(ctrl_keys)} keys (missing={len(c_miss)}, unexpected={len(c_unexp)})")
            if ema_ctrl is not None:
                e_miss, e_unexp = ema_ctrl.load_state_dict(ctrl_keys, strict=False)
                logger.info(f"[resume] ema_ctrl_encoder: {len(ctrl_keys)} keys (missing={len(e_miss)}, unexpected={len(e_unexp)})")
        # Load optimizer state
        if ck.get("optimizer"):
            try:
                optimizer.load_state_dict(ck["optimizer"])
                logger.info("[resume] optimizer state loaded")
            except Exception as e:
                logger.warning(f"[resume] optimizer load failed: {e}")
        # Load scheduler state
        if ck.get("scheduler"):
            try:
                scheduler.load_state_dict(ck["scheduler"])
                logger.info("[resume] scheduler state loaded")
            except Exception as e:
                logger.warning(f"[resume] scheduler load failed: {e}")
        resume_step = int(ck.get("train_steps", 0) or 0)
        logger.info(f"[resume] resuming from step {resume_step}")
        # Move ctrl back to device (load_state_dict may put tensors on cpu)
        ctrl.to(device)
        if ema_ctrl:
            ema_ctrl.to(device)

    # ---- In-process GPU eval setup (uses this process's GPU memory) ----
    gpu_eval_cache = None
    gpu_eval_vae = None
    if args.gpu_eval_csv and os.path.exists(args.gpu_eval_csv):
        from diffusers.models import AutoencoderKL
        gpu_eval_vae = AutoencoderKL.from_pretrained(
            "pretrained_models/sd-vae-ft-ema").to(device).eval()
        latent_spatial = 32  # DiT-S/2 latent 4x32x32
        gpu_eval_cache = prepare_ctrl_eval_cache(
            args.gpu_eval_csv, args.gpu_eval_img_root, args.gpu_eval_skel_root,
            256, args.gpu_eval_n, 8, 4, 0.18215)
        logger.info(f"[gpu-eval] cache ready: {args.gpu_eval_csv} n={args.gpu_eval_n} "
                    f"(eval every {args.gpu_eval_every} steps)")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(args.results_dir, f"{ts}-{args.experiment_name}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # 供 auto_eval_cpu 轮询定位当前活动实验
    with open(os.path.join(args.results_dir, "_active_ckpt_dir.txt"), "w", encoding="utf-8") as _m:
        _m.write(ckpt_dir)
    logger.info(f"results: {exp_dir}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0, prefetch_factor=4)

    step = resume_step
    t0 = time.time()
    log_steps = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
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

            # 统一时间步采样: FlowMatching.sample_t -> t∈[0,1); GaussianDiffusion.sample_t -> t∈{0..T-1}。
            # 调用方绝不自己分支 (否则会重蹈 flow/randint 错配覆辙)。
            t = diffusion.sample_t(x_latent.shape[0], device)
            model_kwargs = dict(y_callig=y_callig, y_char=y_char, cond=skel)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(ctrl, x_latent, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            if not torch.isfinite(loss):
                logger.warning(f"[nan] step {step}; skip")
                del loss_dict, loss
                continue

            # Capture scalar before del
            _v = loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # === INFRA: free the graph every step ===
            del loss, loss_dict

            if ema_ctrl is not None:
                if step < 2000:
                    ema_decay = 0.9999 * (1 - (1 - step / 2000) ** 4)
                else:
                    ema_decay = args.ema_decay
                update_ema(ema_ctrl, ctrl, ema_decay)

            running_loss += _v
            log_steps += 1
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                sps = log_steps / max(dt, 1e-6)
                mem_r = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"(step={step:07d}) loss={running_loss/log_steps:.4f} | "
                            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                            f"Steps/Sec: {sps:.2f} | Mem: {mem_r:.2f}G")
                running_loss = 0.0
                log_steps = 0
                t0 = time.time()

            if step % args.ckpt_every == 0 and step > 0:
                # Save trainable weights (ctrl_encoder always; + main if from-scratch)
                ck = {
                    "ctrl": {k: v.detach().cpu() for k, v in ctrl.state_dict().items()
                             if k.startswith("ctrl_encoder")},
                    "ema": {k: v.detach().cpu() for k, v in ema_ctrl.state_dict().items()
                            if k.startswith("ctrl_encoder")} if ema_ctrl else None,
                    "train_steps": step,
                    "args": vars(args),
                    "saved_at": datetime.datetime.now().isoformat(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if not args.train_ctrl_only:
                    # from-scratch: also save full model weights
                    ck["model"] = {k: v.detach().cpu() for k, v in ctrl.state_dict().items()
                                   if k.startswith("main.")}
                    if ema_ctrl:
                        ck["ema_model"] = {k: v.detach().cpu() for k, v in ema_ctrl.state_dict().items()
                                           if k.startswith("main.")}
                torch.save(ck, os.path.join(ckpt_dir, f"{step:07d}.pt"))
                # 写 .done 标记, 供 auto_eval_cpu 确认 ckpt 写完整
                open(os.path.join(ckpt_dir, f"{step:07d}.pt") + ".done", "w").close()
                logger.info(f"[save] {step}")

                # In-process GPU eval (base vs GT-skel) -> pending marker for CPU daemon
                if gpu_eval_cache is not None and step % args.gpu_eval_every == 0:
                    _eval_model = ema_ctrl if ema_ctrl is not None else ctrl
                    try:
                        run_ctrl_pair_eval(
                            _eval_model, gpu_eval_vae, diffusion, gpu_eval_cache,
                            device, step, ckpt_dir,
                            ddim_steps=args.gpu_eval_steps, cfg_scale=args.gpu_eval_cfg,
                            dit_batch=args.gpu_eval_dit_batch,
                            vae_batch=args.gpu_eval_vae_batch)
                    except Exception as _e:
                        logger.warning(f"[gpu-eval] step {step} FAILED: {_e}")

                if args.ckpt_keep > 0:
                    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
                    for p in pts[:-args.ckpt_keep]:
                        os.remove(p)
                        # 同步删 .done 标记
                        if os.path.exists(p + ".done"):
                            os.remove(p + ".done")

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    logger.info("Done!")


if __name__ == "__main__":
    main()
