# -*- coding: utf-8 -*-
"""
train_controlnet.py — 训练 ControlNet (latent DiT + skel VAE-latent 条件).

两种模式:
  A) warm-start (默认, train_ctrl_only=True): 冻结已训练主模型, 只训练 ctrl_encoder
  B) from-scratch (train_ctrl_only=False): 主模型+ctrl_encoder 一起从零训练

skel 条件:
  * 3px skel 图 → VAE encode → latent (4,32,32), 存 latent shards (与主 latent 同格式)
  * 训练时从 shard 加载, 与主模型 x latent 同空间 (不再是 pixel 输入)

流程:
  1. 构建 DiT_2Cond-S/2 + ControlNetDiT 包装
  2. warm-start: 加载已训练主模型 ckpt 并冻结; from-scratch: 可选加载 pretrained body
  3. 数据: latent shards + skel latent shards
  4. 训练: flow velocity / ddpm-eps loss, cond=skel latent (4,32,32)
  5. 零注入 → 完美 warm-start (初始 ctrl=0, 主模型行为不变)
  6. 结构条件 dropout (10%): 让模型学到无 skel 也能生成 (CFG 友好)

INFRA 设计:
  - warm-start: 主模型冻结, forward 不建训练图 → 只有 ctrl_encoder 建图
  - 每步 del 所有中间张量 + zero_grad → 无 graph 残留
  - 不加载 VAE (latent mode) → 省 ~500MB

用法:
  python -m src.train.train_controlnet --config src/train/configs/ctrl_skel_s19_flow.json
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
import re
import logging
import datetime
import argparse
import json

import torch
import torch.nn as nn
import numpy as np

from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow, flow_kwargs_from
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
    """EMA 只更新 trainable 参数 (requires_grad=True). frozen 参数直接 copy (保持一致).

    用 ``named_parameters()`` 做**名字**匹配，而不是 ``zip(parameters())`` 做位置匹配：
    zip 在两侧长度不一致时会静默截断，一旦有人把 ema 换成"结构相同但重建的
    模型"就会静默错位，且不会有任何报错。这里与 train.py 的主训练循环保持一致。
    """
    with torch.no_grad():
        src = dict(model.named_parameters())
        miss = 0
        for name, ep in ema_model.named_parameters():
            p = src.get(name)
            if p is None:
                miss += 1
                continue
            if p.requires_grad:
                ep.data.mul_(decay).add_(p.data, alpha=1 - decay)
            else:
                ep.data.copy_(p.data)
        if miss:
            logger.warning(f"[ema] {miss} ema params not found in model (arch mismatch?)")
        src_b = dict(model.named_buffers())
        for name, eb in ema_model.named_buffers():
            b = src_b.get(name)
            if b is not None and eb.shape == b.shape:
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
    ap.add_argument("--skel-root", default="final_skeleton_1px",
                    help="(兼容/显示用) 1px skel PNG 根目录; 训练条件优先用 skel_latent_shards_dir")
    ap.add_argument("--skel-latent-shards-dir", default="final_skel_latents_fame_1px",
                    help="skel VAE latent shards 目录 (ControlNet 条件, 4ch/32x32). "
                         "为空则退回 skel-root PNG (旧行为). 默认 1px latent")
    ap.add_argument("--blacklist-csv", default="",
                    help="GT 噪点审计 blacklist (img_id,reasons), 命中的样本在训练中被过滤. "
                         "空则不过滤")
    ap.add_argument("--results-dir", default="5script/results/ctrl_skel")
    ap.add_argument("--experiment-name", default="ctrl-skel-1px")
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
    # ---- IDS 组件码本字嵌入 (主模型用 IDS 时必须传) ----
    ap.add_argument("--use-ids-char-embedder", type=_str_to_bool, default=False,
                    help="主模型是否用 IDS 组件码本字嵌入 (s25 及以后). 加载主模型时必须与训练一致")
    ap.add_argument("--ids-file", default="",
                    help="IDS 字典文件路径 (cjkvi ids.txt)")
    ap.add_argument("--use-std-dino-char-embedder", type=_str_to_bool, default=False,
                    help="主模型是否用标准字形 DINO 冻结表字嵌入 (s28). 加载主模型时必须与训练一致")
    ap.add_argument("--std-dino-table-path", default="",
                    help="标准字形 DINO 表路径 (默认 _sync_work/std_dino_char_table_384_pca.npy)")
    ap.add_argument("--ids-char-map-csv", default="",
                    help="含 character_id,character 列的 csv, 用于 char_id->char 映射. "
                         "空则假设 char_id==Unicode codepoint")
    ap.add_argument("--diffusion-type", default="ddpm", choices=["ddpm", "flow"],
                    help="main model diffusion type: ddpm (eps/DDIM) or flow (velocity/ODE)")

    # ---- Flow Matching 配置（必须与主模型训练时一致，否则训练/推理不一致）----
    ap.add_argument("--t-sampler", default="logit_normal",
                    choices=["uniform", "logit_normal", "cosmap"], dest="t_sampler")
    ap.add_argument("--t-mean", type=float, default=0.0, dest="t_mean")
    ap.add_argument("--t-std", type=float, default=1.0, dest="t_std")
    ap.add_argument("--flow-sampler", default="heun", choices=["euler", "heun"],
                    dest="flow_sampler")
    ap.add_argument("--flow-heun-batch", type=int, default=1, dest="heun_batch")
    ap.add_argument("--flow-shift", type=float, default=1.0, dest="shift")
    ap.add_argument("--learn-sigma", type=int, default=None, choices=[0, 1],
                    help="None=auto (flow->False, ddpm->True)")

    # ---- 骨干现代化（必须与主模型训练时一致）----
    ap.add_argument("--norm-type", default="rms", choices=["rms", "layer"], dest="norm_type")
    ap.add_argument("--mlp-type", default="swiglu", choices=["swiglu", "gelu"], dest="mlp_type")
    ap.add_argument("--qk-norm", type=int, default=1, choices=[0, 1], dest="qk_norm")
    ap.add_argument("--rope", type=int, default=1, choices=[0, 1], dest="rope")
    ap.add_argument("--rope-theta", type=float, default=100.0, dest="rope_theta")
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "xformers", "eager"], dest="attn_impl")

    # ---- torch.compile (torch>=2.0, cu121 env) ----
    ap.add_argument("--compile", type=_str_to_bool, default=False,
                    help="Wrap entire ControlNetDiT (ctrl encoder + frozen main) with "
                         "torch.compile before optimizer setup (needs torch>=2.0).")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode.")

    # ---- ctrl 早停 (基于 gpu_eval 的 ctrl.ssim, 与 base train.py 语义一致) ----
    ap.add_argument("--early-stop", type=_str_to_bool, default=False,
                    help="Early-stop ctrl training when gpu_eval ctrl.ssim stops improving "
                         "for --early-stop-patience consecutive evals.")
    ap.add_argument("--early-stop-metric", default="ssim",
                    choices=["ssim", "mse"],
                    help="Metric driving early stop. ssim=higher better; mse=lower better.")
    ap.add_argument("--early-stop-patience", type=int, default=5,
                    help="Stop after N consecutive evals without improvement.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.002,
                    help="Min improvement to count as NEW BEST (avoids noise-driven restart).")
    ap.add_argument("--early-stop-min-steps", type=int, default=0,
                    help="Do not early-stop before this many total steps.")

    # ---- ControlNet 结构 ----
    ap.add_argument("--ctrl-depth", type=int, default=0,
                    help="ctrl encoder 深度; 0=与主模型同深。设为主模型深度的一半可省一半参数")
    ap.add_argument("--ctrl-hidden", type=int, default=0,
                    help="ctrl encoder 宽度; 0=与主模型同宽")
    ap.add_argument("--ctrl-num-heads", type=int, default=0,
                    help="ctrl encoder 注意力头数; 0=与主模型一致")
    ap.add_argument("--injection", default="modulate", choices=["modulate", "add"],
                    help="注入方式: modulate = x*(1+s)+t (可增强也可抑制); add = x+feat (旧)")
    ap.add_argument("--null-cond", default="gaussian", choices=["gaussian", "zeros", "learned"],
                    help="cond dropout 时的替代条件。zeros(旧)=解码成特定灰块, 与真实骨骼"
                         "分布差距大; gaussian 更接近无信息先验")
    ap.add_argument("--cond-drop-all-prob", type=float, default=0.05)
    ap.add_argument("--cond-drop-one-prob", type=float, default=0.25)
    ap.add_argument("--cond-drop-struct-prob", type=float, default=0.1,
                    help="skel 条件随机置零概率 (训练时), 让模型学到无 skel 也能生成")
    ap.add_argument("--cond-drop-which-glyph-prob", type=float, default=0.5,
                    help="与主模型训练时一致的 glyph 条件丢弃概率 (s20 base=0.75)",
                    dest="cond_drop_which_glyph_prob")
    ap.add_argument("--skel-cond-channels", type=int, default=4)
    # ---- B 段早期 REPA 挂载 (可选, 默认关闭) ----
    # REPA 对齐主模型中间特征到 DINOv2。warm-start 下主模型冻结, REPA 梯度只经
    # ctrl 注入路径回传 (少量锚定, 不破坏主模型); from-scratch 下完全生效。
    ap.add_argument("--w-repa-early", type=float, default=0.0,
                    help="skel-ctrl 训练期 REPA 权重 (0=关闭; 建议 0.05~0.1 渐进)")
    ap.add_argument("--repa-early-layer", type=int, default=8,
                    help="早期 REPA 对齐的主模型 block 层 (默认 8)")
    ap.add_argument("--repa-early-warmup", type=int, default=2000,
                    help="REPA 权重线性从 0 爬到 --w-repa-early 的步数")
    ap.add_argument("--img-root", default="final_imgs_fame_v8",
                    help="GT 图根目录 (w_repa_early>0 时加载, REPA 教师输入)")
    ap.add_argument("--repa-teacher-ckpt", type=str, default="",
                    help="本地 DINOv2 safetensors 教师; 空=自动查找")
    ap.add_argument("--unfreeze-main", type=_str_to_bool, default=False,
                    help="warm-start 后解冻主模型 (除 char 表), 用 --main-lr 低学习率联合训练")
    ap.add_argument("--main-lr", type=float, default=3e-5,
                    help="unfreeze-main 时主模型的独立低学习率")
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
    ap.add_argument("--gpu-eval-skel-root", default="final_skel1_fame")
    ap.add_argument("--gpu-eval-skel-latent-shards-dir", default="",
                    help="eval 用 skel VAE latent shards (与训练条件一致); 空=PNG")
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

    # ---- 数据: latent shards + skel latent (或 3px skel PNG) ----
    use_skel_latent = bool(args.skel_latent_shards_dir)
    _repa_early = bool(getattr(args, 'w_repa_early', 0) or 0) > 0
    logger.info(f"[data] loading latent + skel({'latent' if use_skel_latent else 'png'})"
                f"{' + image(REPA)' if _repa_early else ''} ...")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        img_root=args.img_root if _repa_early else None,
        skel_root=args.skel_root,
        skel_latent_shards_dir=args.skel_latent_shards_dir,
        image_size=256, load_canny=False, load_skel=not use_skel_latent,
        is_train=True, preload=bool(args.preload), load_image=_repa_early,
        num_preload_workers=args.preload_workers, structure_size=256)
    n = len(ds)
    logger.info(f"[data] {n} samples, skel from "
                f"{args.skel_latent_shards_dir or args.skel_root}")

    # arch 参数：主模型与 ctrl encoder 必须用同一组，否则两侧特征分布不匹配
    _arch = dict(
        norm_type=args.norm_type, mlp_type=args.mlp_type,
        qk_norm=bool(args.qk_norm), rope=bool(args.rope),
        rope_theta=args.rope_theta, attn_impl=args.attn_impl)
    _ls = getattr(args, 'learn_sigma', None)
    _learn_sigma = bool(_ls) if _ls is not None else \
        str(args.diffusion_type).lower() not in ("flow", "flow_matching", "fm")
    _ctrl_cfg = dict(
        ctrl_depth=(args.ctrl_depth or None), ctrl_hidden=(args.ctrl_hidden or None),
        ctrl_num_heads=(args.ctrl_num_heads or None),
        injection=args.injection, null_cond=args.null_cond)
    logger.info(f"[arch] {_arch} | learn_sigma={_learn_sigma} | ctrl={_ctrl_cfg}")

    # ---- 模型构建 ----
    if args.train_ctrl_only:
        # warm-start: 加载已训练主模型, 冻结
        # 注意：不再做 `os.path.exists(...) or None` 兜底 —— 路径失效必须硬失败，
        # 否则会用随机初始化的主模型训练几十小时。
        logger.info("[model] warm-start: loading main model + freezing ...")
        # IDS 组件码本: 构建 char_id -> char 映射 (主模型为 IDS 时)
        _ids_char_id_to_char = None
        if getattr(args, 'use_ids_char_embedder', False):
            _ids_csv = getattr(args, 'ids_char_map_csv', '')
            if _ids_csv and os.path.isfile(_ids_csv):
                from src.model.ids_embedder import build_char_id_map_from_csv
                _ids_char_id_to_char = build_char_id_map_from_csv(_ids_csv)
                logger.info(f"[ids] char_id->char map from {_ids_csv}: "
                            f"{len(_ids_char_id_to_char)} entries")
            else:
                logger.warning("[ids] ids_char_map_csv not found, assuming char_id==Unicode")

        main_model = load_main_model(
            model_name=args.model, ckpt_path=args.main_ckpt,
            device=device, num_calligraphers=args.num_calligraphers, num_characters=args.num_characters,
            condition_fusion=args.condition_fusion, callig_embed_dim=args.callig_embed_dim,
            char_embed_dim=args.char_embed_dim, char_proj_mode=args.char_proj_mode,
            freeze_char_table=args.freeze_char_table,
            use_ids_char_embedder=getattr(args, 'use_ids_char_embedder', False),
            ids_file=getattr(args, 'ids_file', '') or None,
            char_id_to_char=_ids_char_id_to_char,
            use_std_dino_char_embedder=getattr(args, 'use_std_dino_char_embedder', False),
            std_dino_table_path=getattr(args, 'std_dino_table_path', '') or None,
            cond_drop_all_prob=args.cond_drop_all_prob,
            cond_drop_one_prob=args.cond_drop_one_prob,
            cond_drop_which_glyph_prob=args.cond_drop_which_glyph_prob,
            use_checkpoint=args.use_checkpoint,
            learn_sigma=_learn_sigma, diffusion_type=args.diffusion_type, **_arch)
        main_model.eval()
        ctrl = ControlNetDiT(main_model, cond_in_channels=args.skel_cond_channels,
                            train_ctrl_only=True, **_ctrl_cfg, **_arch).to(device)
    else:
        # from-scratch: 构建新主模型, 可选加载 pretrained body
        logger.info("[model] from-scratch: building new main model ...")
        latent_size = 32  # DiT-S/2 latent
        main_model = DiT_2Cond_models[args.model](
            num_calligraphers=args.num_calligraphers, num_characters=args.num_characters,
            condition_fusion=args.condition_fusion,
            callig_embed_dim=args.callig_embed_dim, char_embed_dim=args.char_embed_dim,
            cond_drop_all_prob=args.cond_drop_all_prob, cond_drop_one_prob=args.cond_drop_one_prob,
            cond_drop_which_glyph_prob=args.cond_drop_which_glyph_prob,
            use_checkpoint=args.use_checkpoint, learn_sigma=_learn_sigma, **_arch)
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
                            train_ctrl_only=False, **_ctrl_cfg, **_arch).to(device)

    trainable = [p for p in ctrl.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in ctrl.parameters() if not p.requires_grad)
    logger.info(f"[ctrl] trainable params: {n_train:,} | frozen: {n_frozen:,}")

    # ---- 可选: 解冻主模型 (联合训练, 除 char 表) ----
    # warm-start 分支里 load_main_model 已 freeze_char_table 冻结 char 表,
    # 其余主模型参数本来 requires_grad=False (train_ctrl_only=True 语义)。
    # 解冻 = 把这些参数置 True 并加入 trainable, optimizer 用两个 param group
    # (ctrl 组 lr=args.lr, 主模型组 lr=args.main_lr)。
    _main_lr_group = None
    _main_params = []
    if getattr(args, "unfreeze_main", False):
        main_p = [p for p in ctrl.main.parameters() if not p.requires_grad]
        for p in main_p:
            p.requires_grad = True
        trainable += main_p
        _main_lr_group = "main"
        _main_params = main_p
        n_main = sum(p.numel() for p in main_p)
        logger.info(f"[unfreeze-main] 主模型 {n_main:,} 参数解冻 (lr={args.main_lr}); "
                    f"总 trainable {sum(p.numel() for p in trainable):,}")

    # torch.compile (torch>=2.0, cu121 env): 在 EMA deepcopy 与 optimizer 之前编译
    # 整个 ControlNetDiT（ctrl encoder + 冻结主模型）。EMA 是 eval-only 深拷贝，
    # 不编译（省编译时间/显存），权重同步走普通张量拷贝，与编译无关。
    if getattr(args, "compile", False):
        if float(torch.__version__.split("+")[0][:3]) < 2.0:
            logger.warning("[compile] torch %s 不支持 torch.compile，忽略 --compile",
                           torch.__version__)
        else:
            logger.info(f"[compile] torch.compile(mode={args.compile_mode}) 注入 ...")
            ctrl = torch.compile(ctrl, mode=args.compile_mode)

    # EMA on trainable params (ctrl_encoder + optionally main_model)
    ema_ctrl = None
    if args.use_ema:
        ema_ctrl = copy.deepcopy(ctrl).eval()
        requires_grad(ema_ctrl, False)

    # ---- B 段早期 REPA (可选) ----
    repa_early = None
    if _repa_early:
        from src.loss import REPALoss
        student_dim = 384  # DiT-S/2 block 输出维度
        repa_early = REPALoss(
            student_dim=student_dim, teacher_backbone="dinov2_vits14",
            teacher_ckpt=args.repa_teacher_ckpt or None).to(device).eval()
        for p in repa_early.proj.parameters():
            p.requires_grad = True
        trainable += [p for p in repa_early.proj.parameters() if p.requires_grad]
        logger.info(f"[repa-early] w={args.w_repa_early} layer={args.repa_early_layer} "
                    f"warmup={args.repa_early_warmup} proj_trainable={sum(p.numel() for p in repa_early.proj.parameters()):,}")

    # flow 的 t 分布 / 求解器配置与 train.py 共用同一份（否则训练/推理不一致）
    diffusion = create_diffusion_or_flow(
        timestep_respacing="", diffusion_type=args.diffusion_type,
        **flow_kwargs_from(args))
    if getattr(diffusion, 'is_flow', False):
        logger.info(f"[flow] {diffusion.describe()}")

    if _main_lr_group:
        # 两个 param group: ctrl 组 (args.lr) + 主模型组 (args.main_lr)
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in trainable if p not in _main_params]},
                {"params": _main_params, "lr": args.main_lr},
            ],
            lr=args.lr, weight_decay=args.weight_decay)
    else:
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
            # 非 main.* 全载 (与存盘侧对称; 兼容旧 ckpt —— 旧 ckpt 本来就只有 ctrl_encoder.*)
            ctrl_keys = {k: v for k, v in ctrl_src.items() if not k.startswith("main.")}
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
            256, args.gpu_eval_n, 8, 4, 0.18215,
            skel_latent_shards_dir=args.gpu_eval_skel_latent_shards_dir)
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
    # ctrl 早停状态
    _es_best = None
    _es_stale = 0
    _es_delta = float(getattr(args, 'early_stop_min_delta', 0.002) or 0.002)
    if getattr(args, 'early_stop', False):
        logger.info(f"[early-stop] enabled: metric=ctrl.{args.early_stop_metric}, "
                    f"patience={args.early_stop_patience}, min_delta={_es_delta}, "
                    f"min_steps={getattr(args, 'early_stop_min_steps', 0)}")

    for epoch in range(args.epochs):
        for batch in loader:
            x_latent = batch['latent'].to(device)
            y_callig = batch['y_callig'].to(device)
            y_char = batch['y_char'].to(device)
            # skel 条件: latent 优先 (4,32,32); 否则退回 PNG (1,256,256) 旧行为
            skel = batch['skel_latent'] if batch['skel_latent'].numel() else batch['skeleton']
            skel = skel.to(device).float()

            # skel 条件 dropout。
            # 注意：不再一律置零 —— 零 latent 解码后是"特定灰色块"而不是空白，
            # 与真实骨骼分布差距很大，会让 CFG 的 uncond 分支落在分布外。
            # 默认用 gaussian（更接近无信息先验）；--null-cond 可改回 zeros。
            if args.cond_drop_struct_prob > 0:
                drop = torch.rand(x_latent.shape[0], device=device) < args.cond_drop_struct_prob
                if drop.any():
                    null = ctrl._make_null(skel)
                    skel = torch.where(drop.view(-1, 1, 1, 1).expand_as(skel), null, skel)

            # 统一时间步采样: FlowMatching.sample_t -> t∈[0,1); GaussianDiffusion.sample_t -> t∈{0..T-1}。
            # 调用方绝不自己分支 (否则会重蹈 flow/randint 错配覆辙)。
            t = diffusion.sample_t(x_latent.shape[0], device)
            model_kwargs = dict(y_callig=y_callig, y_char=y_char, cond=skel)
            if repa_early is not None:
                model_kwargs['return_intermediate_layer'] = args.repa_early_layer

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(ctrl, x_latent, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                # B 段早期 REPA: 对齐主模型 block 特征到 DINOv2 (w 线性渐进)
                loss_repa = torch.tensor(0.0, device=device)
                if repa_early is not None:
                    int_feats = loss_dict.get("intermediate_feats", None)
                    img_gt = batch.get('image')
                    if int_feats is not None and img_gt is not None and img_gt.numel() > 0:
                        wr = args.w_repa_early * min(
                            1.0, step / max(args.repa_early_warmup, 1))
                        loss_repa = repa_early(int_feats, img_gt.to(device))
                        loss = loss + wr * loss_repa

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
                # EMA warmup：早期用更小的 decay 让 EMA 快速追上仍在快速变化的权重。
                # 注意 base 用 args.ema_decay（旧代码硬编码 0.9999，导致
                # --ema-decay 在 warmup 结束后才生效，与 train.py 的策略也不一致）。
                _ema_base = min(float(args.ema_decay), 0.9999)
                if step < 2000:
                    ema_decay = _ema_base * (1 - (1 - step / 2000) ** 4)
                else:
                    ema_decay = _ema_base
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
                    # 非 main.* 全存: ctrl_encoder + injections (zero-conv/modulate 注入)
                    # 旧过滤器只存 ctrl_encoder.* 会把训练好的 injections 丢掉
                    "ctrl": {k: v.detach().cpu() for k, v in ctrl.state_dict().items()
                             if not k.startswith("main.")},
                    "ema": {k: v.detach().cpu() for k, v in ema_ctrl.state_dict().items()
                            if not k.startswith("main.")} if ema_ctrl else None,
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

                # ---- ctrl 早停: 读 daemon 写的 eval_auto_ctrl_*.json (ctrl.ssim 越高越好) ----
                if (getattr(args, 'early_stop', False) and gpu_eval_cache is not None
                        and step >= int(getattr(args, 'early_stop_min_steps', 0))
                        and step % args.gpu_eval_every == 0):
                    _ev_files = sorted(glob.glob(os.path.join(ckpt_dir, "eval_auto_ctrl_*.json")))
                    _ev_last = _ev_files[-1] if _ev_files else None
                    if _ev_last:
                        try:
                            with open(_ev_last, "r", encoding="utf-8") as _f:
                                _d = json.load(_f)
                            _ctrl_res = _d.get("ctrl", _d)
                            _metric = (args.early_stop_metric or "ssim").lower()
                            _val = _ctrl_res.get(_metric)
                            _higher = _metric == "ssim"
                            if _val is not None:
                                if (_es_best is None or
                                        (_val > _es_best + _es_delta if _higher
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
                                        step = args.max_steps
                        except Exception as _ee:
                            logger.warning(f"[early-stop] read {_ev_last} failed: {_ee}")

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
