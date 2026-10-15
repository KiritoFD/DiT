# -*- coding: utf-8 -*-
"""setup.py — 训练**运行时装配**（2026-10-13 [审查 PR4] 从 src/train/train.py 迁出）

审查 PR4 目标结构: `train/setup.py  # 分布式、随机种子、目录、device、logging` ✓

含两块:
  · `create_logger(logging_dir)` —— 从 train.py 原样搬入 ✓（logging 属于 setup ✓）
  · `setup_runtime(args) -> RuntimeSetup` —— DDP 初始化 / 种子 / 目录 / device / logger ✓

⚠ 语义保真: `experiment_dir` / `checkpoint_dir` **只在 rank 0 存在** ✓（原实现即如此 ✓，
   它们在 rank 0 守卫内被使用 ✓）。本模块对 rank != 0 显式返回 `None` ✓ ——
   把原"误用即 NameError"的隐式约束**显式化** ✓，对所有**可达**路径等价 ✓✓。

与原实现**逐行等价** ✓（包成函数 + 返回 RuntimeSetup；注释原样保留 ✓）。
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist

__all__ = ["RuntimeSetup", "create_logger", "setup_runtime"]


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@dataclass(frozen=True)
class RuntimeSetup:
    """进程级初始化结果 ✓（rank != 0 时 experiment_dir/checkpoint_dir 为 None ✓）。"""
    rank: int
    device: int
    seed: int
    logger: Any
    experiment_dir: Optional[str]
    checkpoint_dir: Optional[str]
    vae_downscale: int
    latent_size: int


def setup_runtime(args):
    """完成进程级初始化并返回 RuntimeSetup ✓（**逐行等价**于 train.py 原块 ✓）。"""
    backend = "nccl" if (dist.is_nccl_available() and sys.platform != "win32") else "gloo"
    dist.init_process_group(backend)
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    
    experiment_dir = None
    checkpoint_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.model.replace("/", "-")
        # Timestamp-named experiment dir (unique per launch, never collides or overwrites).
        # 命名规范 (跟随 assets/results/): results_dir 按**实验**独立, 其下每次
        # 启动建一个时间戳 run 目录 —— 同名实验多次启动互不覆盖, 跨实验不冲突。
        _ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        _name = getattr(args, "experiment_name", "") or model_string_name
        _name = re.sub(r"[^A-Za-z0-9._-]+", "-", _name).strip("-")
        experiment_dir = f"{args.results_dir}/{_ts}-{_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 供外部 CPU 评测进程定位当前活动实验的 ckpt 目录。
        with open(f"{args.results_dir}/_active_ckpt_dir.txt", "w", encoding="utf-8") as _m:
            _m.write(checkpoint_dir + "\n")
        # log.txt lives inside this experiment dir (created first), never overwritten.
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # ★ 评测产物 (summary/batch csv, posters/) 默认落**本次 run 目录**:
        #   每次启动都是新目录 -> 与历史/其他实验的 (step,set) 永不撞车, 评测
        #   不会被去重逻辑静默跳过。--eval-dir 可显式覆盖 (如汇总到别的盘)。
        args.eval_dir = (str(getattr(args, "eval_dir", "") or "").strip()
                         or experiment_dir)
        os.makedirs(args.eval_dir, exist_ok=True)
        logger.info(f"[eval-dir] eval artifacts -> {args.eval_dir}")
        with open(f"{experiment_dir}/resolved_config.json", "w", encoding="utf-8") as _cf:
            json.dump(vars(args), _cf, ensure_ascii=False, indent=2)
        _sources = {}
        # ★ 2026-10-13 [审查 P0]: 原块只哈希根目录旧文件名(实际为空 ✗) -> 改为自包含模块 ✓
        # ★ 2026-10-14 [重构 P1-7]: 扩展为**完整依赖清单** ✗
        #   ① 代码多根: src + tools ✓（Critic 实现从 tools/ 动态加载 ✗ 只哈希 src 不够 ✗）
        #   ② 依赖身份: 模型权重 / VAE / 字体 / 数据 manifest / 生成器 / 评测器 各自 SHA256 ✓
        #   规则: 配置里**非空**的路径才纳入 ✓（空 = 未启用 ✓ 不算缺失 ✗）；
        #         非空但盘上不存在 ⇒ 显式记入 artifacts_missing ✗（不静默省略 ✗）
        from src.train.manifest import write_source_manifest
        _ART_KEYS = {
            "vae": "vae_path",
            "callig_emb_pretrained": "callig_emb_pretrained",
            "deform_ckpt": "deform_ckpt",
            "critic_ckpt": "critic_ckpt",
            "critic_std_bank": "critic_std_bank",
            "style_rank_ckpt": "style_rank_ckpt",
            "style_rank_protos": "style_rank_protos",
            "style_rank_cent": "style_rank_cent",
            "train_csv": "csv",
            "eval_csv": "eval_csv",
            "std_soft_root": "std_soft_root",
            "repa_cache_dir": "repa_cache_dir",
            "latent_shards": "shards",
            "aux_latent_shards_dirs": "aux_latent_shards_dirs",
            "fonts_dir": None,          # 常量 ✓ 见下
        }
        _arts = {"fonts_dir": "tools/fonts"}          # 字体目录是常量 ✓ 始终纳入 ✓
        for _nm, _attr in _ART_KEYS.items():
            if _attr is None:
                continue
            _v = getattr(args, _attr, None)
            if isinstance(_v, (list, tuple)):
                _v = ",".join(str(x) for x in _v if str(x).strip())
            _v = str(_v or "").strip()
            if _v:                                     # 非空才纳入 ✓
                _arts[_nm] = _v
        write_source_manifest(experiment_dir,
                              extra={"config_path": getattr(args, "config", None)},
                              artifacts=_arts)
    else:
        logger = create_logger(None)

    # Latent spatial size: default f8 (sd-vae), but supports f4 (kl-f4) via vae_downscale arg.
    vae_downscale = getattr(args, 'vae_downscale', 8)
    assert args.image_size % vae_downscale == 0, f"image_size {args.image_size} not divisible by vae_downscale {vae_downscale}"
    latent_size = args.image_size // vae_downscale
    return RuntimeSetup(rank=rank, device=device, seed=seed, logger=logger,
                        experiment_dir=experiment_dir, checkpoint_dir=checkpoint_dir,
                        vae_downscale=vae_downscale, latent_size=latent_size)
