# -*- coding: utf-8 -*-
"""
train_repa.py — REPA 联合微调 (pipeline 阶段C, 在 skel-ctrl 收敛后运行)

输入:
  --main-ckpt : S30 base ckpt (主模型权重, 与 s31 ctrl 训练时一致)
  --ctrl-ckpt : s31 skel-ctrl ckpt (ctrl encoder + injections)

联合损失 (避免灾难遗忘):
  L = loss_diff + w_repa * loss_repa
    * loss_diff : flow velocity MSE (维持去噪/骨架能力, 与 skel-ctrl 相同)
    * loss_repa : 主模型 block-8 中间特征对齐 DINOv2 语义特征 (冻结 teacher)

用法:
  python src/train/train_repa.py \\
      --config src/train/configs/s32_repa_finetune.json \\
      --main-ckpt <S30 base ckpt> --ctrl-ckpt <s31 ckpt>
"""
import os
import sys
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _r not in sys.path:
    sys.path.insert(0, _r)
import copy
import glob
import time
import math
import logging
import datetime
import argparse
import json

import torch
import torch.nn as nn
import numpy as np

from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow, flow_kwargs_from, REPALoss
from src.utils import MCCDLatentDataset
from src.model.controlnet import ControlNetDiT, load_main_model
from src.eval.in_process_ctrl_eval import prepare_ctrl_eval_cache, run_ctrl_pair_eval

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format='[\033[34m%(asctime)s\033[0m] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("repa")


def _str_to_bool(v):
    return str(v).lower() in ("true", "1", "yes", "on")


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        src = dict(model.named_parameters())
        for name, ep in ema_model.named_parameters():
            p = src.get(name)
            if p is None:
                continue
            if p.requires_grad:
                ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
            else:
                ep.data.copy_(p.data)
        src_b = dict(model.named_buffers())
        for name, eb in ema_model.named_buffers():
            b = src_b.get(name)
            if b is not None and eb.shape == b.shape:
                eb.data.copy_(b.data)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    # ---- 两个输入 ckpt ----
    ap.add_argument("--main-ckpt", default="", help="S30 base 主模型 ckpt (必填)")
    ap.add_argument("--ctrl-ckpt", default="", help="s31 skel-ctrl ckpt (必填)")
    ap.add_argument("--resume", default="",
                    help="从 REPA ckpt 续训 (加载 main+ctrl+ema+optimizer+scheduler+step)")
    # ---- REPA 参数 ----
    ap.add_argument("--w-repa", type=float, default=0.1,
                    help="REPA loss 权重 (0 = 纯 diff 微调, 仅用于消融)")
    ap.add_argument("--repa-layer", type=int, default=8,
                    help="捕获主模型第几个 block 的输出做对齐 (与 base train.py 默认一致)")
    ap.add_argument("--repa-layers", type=str, default="",
                    help="REPA 多层特征 (逗号列表, 如 '8,11'); 非空时优先于 --repa-layer")
    ap.add_argument("--repa-teacher-ckpt", type=str, default="",
                    help="本地 DINOv2 safetensors 教师; 空=自动查找")
    ap.add_argument("--limit-n", type=int, default=0,
                    help="调试: 只用前 N 个训练样本 (0=全部)")
    # ---- 数据 ----
    ap.add_argument("--csv", default="5script/train_fame_clean.csv")
    ap.add_argument("--latent-shards-dir", default="final_latents_fame_clean")
    ap.add_argument("--img-root", default="final_imgs_256", help="GT 图 (REPA 教师输入)")
    ap.add_argument("--skel-root", default="final_skel1_fame")
    ap.add_argument("--skel-latent-shards-dir", default="final_skel_latents_fame_1px")
    ap.add_argument("--results-dir", default="5script/results/s32_repa_finetune")
    ap.add_argument("--experiment-name", default="s32-repa-finetune")
    # ---- 架构 (必须与 S30/s31 一致) ----
    ap.add_argument("--model", default="DiT-2Cond-S/2")
    ap.add_argument("--num-calligraphers", type=int, default=1013)
    ap.add_argument("--num-characters", type=int, default=35130)
    ap.add_argument("--condition-fusion", default="factorized_add")
    ap.add_argument("--callig-embed-dim", type=int, default=128)
    ap.add_argument("--char-embed-dim", type=int, default=384)
    ap.add_argument("--char-proj-mode", default="mlp")
    ap.add_argument("--freeze-char-table", type=_str_to_bool, default=True)
    ap.add_argument("--cond-drop-all-prob", type=float, default=0.05)
    ap.add_argument("--cond-drop-one-prob", type=float, default=0.30)
    ap.add_argument("--cond-drop-which-glyph-prob", type=float, default=0.85)
    ap.add_argument("--cond-drop-struct-prob", type=float, default=0.1)
    ap.add_argument("--skel-cond-channels", type=int, default=4)
    ap.add_argument("--diffusion-type", default="flow", choices=["ddpm", "flow"])
    ap.add_argument("--t-sampler", default="logit_normal", dest="t_sampler")
    ap.add_argument("--t-mean", type=float, default=0.0, dest="t_mean")
    ap.add_argument("--t-std", type=float, default=1.0, dest="t_std")
    ap.add_argument("--flow-sampler", default="heun", choices=["euler", "heun"], dest="flow_sampler")
    ap.add_argument("--flow-heun-batch", type=int, default=1, dest="heun_batch")
    ap.add_argument("--flow-shift", type=float, default=1.0, dest="shift")
    ap.add_argument("--norm-type", default="rms", dest="norm_type")
    ap.add_argument("--mlp-type", default="swiglu", dest="mlp_type")
    ap.add_argument("--qk-norm", type=int, default=1, dest="qk_norm")
    ap.add_argument("--rope", type=int, default=1, dest="rope")
    ap.add_argument("--rope-theta", type=float, default=100.0, dest="rope_theta")
    ap.add_argument("--attn-impl", default="sdpa", dest="attn_impl")
    ap.add_argument("--ctrl-depth", type=int, default=0)
    ap.add_argument("--ctrl-hidden", type=int, default=0)
    ap.add_argument("--ctrl-num-heads", type=int, default=0)
    ap.add_argument("--injection", default="modulate", choices=["modulate", "add"])
    ap.add_argument("--null-cond", default="gaussian", choices=["gaussian", "zeros", "learned"])
    # ---- 训练 ----
    ap.add_argument("--epochs", type=int, default=100000)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--ckpt-keep", type=int, default=3)
    ap.add_argument("--use-ema", type=_str_to_bool, default=True)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--preload", type=_str_to_bool, default=True)
    ap.add_argument("--preload-workers", type=int, default=16)
    ap.add_argument("--use-checkpoint", type=_str_to_bool, default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=_str_to_bool, default=False,
                    help="torch.compile 整个模型 (cu121, 与 ctrl 一致)")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    # ---- in-process GPU eval (监控 skel 质量不回退) ----
    ap.add_argument("--gpu-eval-csv", default="")
    ap.add_argument("--gpu-eval-every", type=int, default=2500)
    ap.add_argument("--gpu-eval-n", type=int, default=100)
    ap.add_argument("--gpu-eval-steps", type=int, default=50)
    ap.add_argument("--gpu-eval-cfg", type=float, default=1.7)
    ap.add_argument("--gpu-eval-img-root", default="final_imgs_256")
    ap.add_argument("--gpu-eval-skel-root", default="final_skel1_fame")
    ap.add_argument("--gpu-eval-skel-latent-shards-dir", default="")
    ap.add_argument("--gpu-eval-dit-batch", type=int, default=8)
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
    if not args.main_ckpt or not args.ctrl_ckpt:
        raise SystemExit("必须提供 --main-ckpt (S30 base) 和 --ctrl-ckpt (s31 ctrl)")

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(0)

    # ---- 数据: latent + skel latent + GT 图 (REPA 教师输入) ----
    logger.info(f"[data] loading latent + skel-latent + image(REPA-需要 {args.w_repa > 0}) ...")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        img_root=args.img_root, skel_root=args.skel_root,
        skel_latent_shards_dir=args.skel_latent_shards_dir,
        image_size=256, load_canny=False, load_skel=False,
        is_train=True, preload=bool(args.preload),
        load_image=bool(args.w_repa > 0),
        num_preload_workers=args.preload_workers, structure_size=256)
    if args.limit_n > 0:
        ds = torch.utils.data.Subset(ds, range(min(len(ds), args.limit_n)))
    logger.info(f"[data] {len(ds)} samples (limit_n={args.limit_n or 'all'})")
    from torch.utils.data import DataLoader
    _w = args.num_workers if use_cuda else 0
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=_w, pin_memory=use_cuda, drop_last=True,
                        persistent_workers=_w > 0, prefetch_factor=4 if _w > 0 else None)

    # ---- 架构 (与 S30/s31 加载主模型一致) ----
    _arch = dict(norm_type=args.norm_type, mlp_type=args.mlp_type,
                 qk_norm=bool(args.qk_norm), rope=bool(args.rope),
                 rope_theta=args.rope_theta, attn_impl=args.attn_impl)
    _ctrl_cfg = dict(ctrl_depth=(args.ctrl_depth or None),
                     ctrl_hidden=(args.ctrl_hidden or None),
                     ctrl_num_heads=(args.ctrl_num_heads or None),
                     injection=args.injection, null_cond=args.null_cond)

    logger.info(f"[model] loading main from {args.main_ckpt} ...")
    main_model = load_main_model(
        model_name=args.model, ckpt_path=args.main_ckpt, device=device,
        num_calligraphers=args.num_calligraphers, num_characters=args.num_characters,
        condition_fusion=args.condition_fusion,
        callig_embed_dim=args.callig_embed_dim, char_embed_dim=args.char_embed_dim,
        char_proj_mode=args.char_proj_mode, freeze_char_table=args.freeze_char_table,
        cond_drop_all_prob=args.cond_drop_all_prob,
        cond_drop_one_prob=args.cond_drop_one_prob,
        cond_drop_which_glyph_prob=args.cond_drop_which_glyph_prob,
        use_checkpoint=args.use_checkpoint, learn_sigma=None,
        diffusion_type=args.diffusion_type, **_arch)
    # REPA 阶段主模型可训练 (字符表仍冻结)
    main_model.train()
    model = ControlNetDiT(main_model, cond_in_channels=args.skel_cond_channels,
                          train_ctrl_only=False, **_ctrl_cfg, **_arch).to(device)

    # 从 s31 ckpt 加载 ctrl encoder + injections (main.* 已由 main_ckpt 加载)。
    # 防御: compile 存盘的 ctrl keys 可能带 _orig_mod. 前缀, 剥掉再载。
    if args.ctrl_ckpt and os.path.exists(args.ctrl_ckpt):
        ck_ctrl = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
        _src = ck_ctrl.get("ema") or ck_ctrl.get("ctrl") or ck_ctrl
        _ctrl_keys = {}
        for k, v in _src.items():
            if k.startswith("main."):
                continue
            _ctrl_keys[k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k] = v
        _miss, _unexp = model.load_state_dict(_ctrl_keys, strict=False)
        logger.info(f"[model] ctrl loaded from {args.ctrl_ckpt}: "
                    f"ctrl_keys={len(_ctrl_keys)}, missing={len(_miss)} (main.* 忽略), "
                    f"unexpected={len(_unexp)}")
        if len(_unexp):
            logger.warning(f"[model] unexpected(前5): {list(_unexp)[:5]}")
    else:
        if args.limit_n > 0:
            logger.warning("[model] --ctrl-ckpt 缺失 (debug/limit_n>0): ctrl encoder 随机初始化")
        else:
            raise SystemExit(f"--ctrl-ckpt 不存在: {args.ctrl_ckpt}")

    # ---- REPA loss (冻结 teacher + 可训 proj) ----
    repa_loss_fn = None
    repa_loss_fns = []
    if args.w_repa > 0:
        _pe = main_model.x_embedder.proj
        student_dim = int(getattr(_pe, "out_features", getattr(_pe, "out_channels", 384)))
        teacher_ckpt = args.repa_teacher_ckpt or None
        # 多层 REPA: --repa-layers "8,11" 优先; 否则退回 --repa-layer 单层
        repa_layers = args.repa_layers.strip()
        if repa_layers:
            repa_layers = [int(x) for x in repa_layers.split(",") if x.strip()]
        else:
            repa_layers = [int(args.repa_layer)]
        logger.info(f"[repa] REPALoss teacher=dinov2_vits14 ckpt={teacher_ckpt or 'auto'} "
                    f"student_dim={student_dim} w={args.w_repa} layers={repa_layers} "
                    f"(L{len(repa_layers)} multi-layer)")
        # 共享 teacher: 多层只加载一次教师模型
        from src.loss import REPALoss as _REPALoss
        _teacher_obj = None
        for i, _rl in enumerate(repa_layers):
            if i == 0:
                rl = _REPALoss(student_dim=student_dim,
                               teacher_backbone="dinov2_vits14",
                               teacher_ckpt=teacher_ckpt).to(device)
                _teacher_obj = rl.teacher
            else:
                rl = _REPALoss(student_dim=student_dim,
                               teacher_backbone="dinov2_vits14",
                               teacher_ckpt=teacher_ckpt,
                               teacher=_teacher_obj).to(device)
            repa_loss_fns.append(rl)
        repa_loss_fn = repa_loss_fns[0]
        logger.info(f"[repa] 共享 teacher: 为 L{len(repa_layers)} 层加载同一教师模型")
        # 供训练循环读取
        args._repa_layers = repa_layers
    else:
        repa_loss_fn = None

    trainable = [p for p in model.parameters() if p.requires_grad]
    if repa_loss_fn is not None:
        trainable.extend([p for p in repa_loss_fn.proj.parameters() if p.requires_grad])
    n_tr = sum(p.numel() for p in trainable)
    logger.info(f"[trainable] {n_tr:,} params (main 解冻 + ctrl + repa.proj)")

    if use_cuda and getattr(args, "compile", False):
        logger.info(f"[compile] torch.compile(mode={args.compile_mode}) ...")
        model = torch.compile(model, mode=args.compile_mode)

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model).eval()
        requires_grad(ema_model, False)

    diffusion = create_diffusion_or_flow(
        timestep_respacing="", diffusion_type=args.diffusion_type,
        **flow_kwargs_from(args))
    if getattr(diffusion, 'is_flow', False):
        logger.info(f"[flow] {diffusion.describe()}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return max(args.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * prog)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---- Resume: 续训加载 (main/ctrl/ema/optimizer/scheduler/step), 新开 exp 目录 ----
    resume_step = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"[resume] loading {args.resume}")
        _ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # 1) 模型权重 (优先 ema_model/ema, 即 EMA 参数, 续训以 EMA 为起点)
        _w = _ck.get("ema_model") or _ck.get("ema") or _ck.get("model") or _ck.get("ctrl") or _ck
        _miss, _unexp = model.load_state_dict(_w, strict=False)
        logger.info(f"[resume] model loaded: missing={len([k for k in _miss if not k.startswith('main.')])}"
                    f"(main 部分忽略), unexpected={len(_unexp)}")
        # 2) EMA 同步
        if ema_model is not None:
            ema_model.load_state_dict(model.state_dict())
        # 3) optimizer/scheduler/step (结构需与当前一致)
        try:
            if _ck.get("optimizer"):
                optimizer.load_state_dict(_ck["optimizer"])
            if _ck.get("scheduler"):
                scheduler.load_state_dict(_ck["scheduler"])
            resume_step = int(_ck.get("train_steps", 0) or 0)
        except Exception as _e:
            logger.warning(f"[resume] optimizer/scheduler 加载失败, 从 0 开始 LR 计划: {_e}")
            resume_step = 0
        logger.info(f"[resume] resume from step {resume_step}")

    # ---- in-process GPU eval (只监控, 不早停) ----
    gpu_eval_cache = None
    gpu_eval_vae = None
    if use_cuda and args.gpu_eval_csv and os.path.exists(args.gpu_eval_csv):
        from diffusers.models import AutoencoderKL
        gpu_eval_vae = AutoencoderKL.from_pretrained(
            "pretrained_models/sd-vae-ft-ema").to(device).eval()
        gpu_eval_cache = prepare_ctrl_eval_cache(
            args.gpu_eval_csv, args.gpu_eval_img_root, args.gpu_eval_skel_root,
            256, args.gpu_eval_n, 8, 4, 0.18215,
            skel_latent_shards_dir=args.gpu_eval_skel_latent_shards_dir)
        logger.info(f"[gpu-eval] cache ready: n={args.gpu_eval_n}, "
                    f"every {args.gpu_eval_every} steps (monitor only)")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(args.results_dir, f"{ts}-{args.experiment_name}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "_active_ckpt_dir.txt"), "w", encoding="utf-8") as _m:
        _m.write(ckpt_dir)
    logger.info(f"results: {exp_dir}")

    step = resume_step
    t0 = time.time()
    log_steps = 0
    run_diff = run_repa = 0.0
    _es_best = None
    _es_stale = 0

    for epoch in range(args.epochs):
        for batch in loader:
            if step >= args.max_steps:
                break
            x_latent = batch['latent'].to(device)
            y_callig = batch['y_callig'].to(device)
            y_char = batch['y_char'].to(device)
            skel = (batch['skel_latent'] if batch['skel_latent'].numel()
                    else batch['skeleton']).to(device).float()
            img = batch['image'].to(device) if args.w_repa > 0 else None

            # skel 条件 dropout (与 s31 ctrl 一致)
            if args.cond_drop_struct_prob > 0:
                drop = torch.rand(x_latent.shape[0], device=device) < args.cond_drop_struct_prob
                if drop.any():
                    null = model._make_null(skel)
                    skel = torch.where(drop.view(-1, 1, 1, 1).expand_as(skel), null, skel)

            t = diffusion.sample_t(x_latent.shape[0], device)
            model_kwargs = dict(y_callig=y_callig, y_char=y_char, cond=skel)
            repa_layers = getattr(args, '_repa_layers', None)
            if repa_layers is not None:
                if len(repa_layers) > 1:
                    model_kwargs['return_intermediate_layers'] = repa_layers
                else:
                    model_kwargs['return_intermediate_layer'] = repa_layers[0]

            if use_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
                    loss_diff = loss_dict["loss"].mean()
            else:
                loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
                loss_diff = loss_dict["loss"].mean()

            loss_repa = torch.tensor(0.0, device=device)
            if repa_loss_fn is not None and img is not None:
                inter = loss_dict.get("intermediate_feats", None)
                if inter is not None:
                    # REPA-L2: inter 为 tuple(list[层], 单层) 或 tuple(list[层]) 或单张量。
                    # 多层的 list 与 repa_loss_fns 一一对应求平均; 单层用 repa_loss_fns[0]。
                    if isinstance(inter, tuple):
                        inner, _extra = inter, None
                        # 兼容 (feats_tuple, single) 形态
                        if len(inter) == 2 and isinstance(inter[0], (list, tuple)) \
                                and isinstance(inter[1], torch.Tensor):
                            inner, _extra = inter[0], inter[1]
                        if len(repa_loss_fns) > 1 and isinstance(inner, (list, tuple)):
                            _losses = [fn(fi, img) for fn, fi in zip(repa_loss_fns, inner)
                                       if fi is not None]
                            if _losses:
                                loss_repa = sum(_losses) / len(_losses)
                            else:
                                loss_repa = torch.tensor(0.0, device=device)
                                logger.warning(f"[repa] step {step}: 多层特征为空, REPA=0")
                        else:
                            loss_repa = repa_loss_fn(inner, img)
                    else:
                        loss_repa = repa_loss_fn(inter, img)
                else:
                    logger.warning(f"[repa] step {step}: intermediate_feats 缺失, REPA 跳过")
            total = loss_diff + args.w_repa * loss_repa

            if not torch.isfinite(total):
                logger.warning(f"[nan] step {step}; skip")
                del loss_dict, total
                continue

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            del loss_dict, total

            if ema_model is not None:
                _eb = min(float(args.ema_decay), 0.9999)
                if step < 2000:
                    _decay = _eb * (1 - (1 - step / 2000) ** 4)
                else:
                    _decay = _eb
                update_ema(ema_model, model, _decay)

            run_diff += loss_diff.item()
            run_repa += loss_repa.item()
            log_steps += 1
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                sps = log_steps / max(dt, 1e-6)
                mem_r = torch.cuda.memory_reserved() / 1024**3 if use_cuda else 0
                logger.info(f"(step={step:07d}) L={run_diff/log_steps + args.w_repa * run_repa/log_steps:.4f}"
                            f" | Diff: {run_diff/log_steps:.4f}"
                            f" | REPA: {run_repa/log_steps:.4f} x {args.w_repa:.2f}"
                            f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                            f" | Steps/Sec: {sps:.2f} | Mem: {mem_r:.2f}G")
                run_diff = run_repa = 0.0
                log_steps = 0
                t0 = time.time()

            if step % args.ckpt_every == 0 and step > 0:
                ew = ema_model if ema_model is not None else model
                ck = {
                    "ctrl": {k: v.detach().cpu() for k, v in model.state_dict().items()
                             if not k.startswith("main.")},
                    "ema": {k: v.detach().cpu() for k, v in ew.state_dict().items()
                            if not k.startswith("main.")} if ema_model else None,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()
                              if k.startswith("main.")},
                    "ema_model": {k: v.detach().cpu() for k, v in ew.state_dict().items()
                                  if k.startswith("main.")} if ema_model else None,
                    "train_steps": step,
                    "args": vars(args),
                    "saved_at": datetime.datetime.now().isoformat(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(ck, os.path.join(ckpt_dir, f"{step:07d}.pt"))
                open(os.path.join(ckpt_dir, f"{step:07d}.pt") + ".done", "w").close()
                logger.info(f"[save] {step}")

                if gpu_eval_cache is not None and step % args.gpu_eval_every == 0:
                    _ev = ema_model if ema_model is not None else model
                    try:
                        run_ctrl_pair_eval(
                            _ev, gpu_eval_vae, diffusion, gpu_eval_cache,
                            device, step, ckpt_dir,
                            ddim_steps=args.gpu_eval_steps, cfg_scale=args.gpu_eval_cfg,
                            dit_batch=args.gpu_eval_dit_batch,
                            vae_batch=args.gpu_eval_vae_batch)
                    except Exception as _e:
                        logger.warning(f"[gpu-eval] step {step} FAILED: {_e}")

                    # ---- REPA 早停: 读 daemon 写的 eval_auto_ctrl_*.json (ctrl.ssim 越高越好) ----
                    # 2026-09-04: v8e 无早停跑满 80k 导致 ctrl 从 0.7761 回退到 0.762 (过拟合退化),
                    # 补早停避免重演。逻辑同 train_controlnet, 由 max_steps 作硬上限。
                    if (getattr(args, 'early_stop', False) and step >= int(getattr(args, 'early_stop_min_steps', 0))):
                        _ev_files = sorted(glob.glob(os.path.join(ckpt_dir, "eval_auto_ctrl_*.json")))
                        if _ev_files:
                            try:
                                with open(_ev_files[-1], "r", encoding="utf-8") as _f:
                                    _d = json.load(_f)
                                _ctrl_res = _d.get("ctrl", _d)
                                _metric = (getattr(args, 'early_stop_metric', 'ssim') or 'ssim').lower()
                                _val = _ctrl_res.get(_metric)
                                _es_delta = float(getattr(args, 'early_stop_min_delta', 0.002) or 0.002)
                                if _val is not None:
                                    if (_es_best is None or
                                            (_val > _es_best + _es_delta if _metric == 'ssim'
                                             else _val < _es_best - _es_delta)):
                                        _es_best = _val
                                        _es_stale = 0
                                        logger.info(f"[early-stop] eval step {step}: ctrl.{_metric}={_val:.4f} "
                                                    f"-> NEW BEST ({_es_best:.4f})")
                                    else:
                                        _es_stale += 1
                                        logger.info(f"[early-stop] eval step {step}: ctrl.{_metric}={_val:.4f} "
                                                    f"(best {_es_best:.4f}, stale {_es_stale}/"
                                                    f"{args.early_stop_patience})")
                                        if _es_stale >= int(args.early_stop_patience):
                                            logger.info(f"[early-stop] ctrl.{_metric} no improvement for "
                                                        f"{_es_stale} evals; stopping ({step}).")
                                            step = args.max_steps  # 结束训练循环
                            except Exception as _ee:
                                logger.warning(f"[early-stop] read {_ev_files[-1]} failed: {_ee}")

                # 运行时绝不删除任何 ckpt (2026-09-04 修正): best 权重可能在任何 step,
                # ckpt_keep 裁剪会永久丢失复盘所需权重。保留全部 + .done 标记。

        if step >= args.max_steps:
            break

    logger.info("Done!")


if __name__ == "__main__":
    main()