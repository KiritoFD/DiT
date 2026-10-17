# -*- coding: utf-8 -*-
"""data_setup.py — 训练**数据装配**（2026-10-13 [审查 PR4] 从 src/train/train.py 迁出）

审查 PR4 目标结构: `train/data_setup.py  # Dataset、sampler、DataLoader` ✓

职责: 由 `args` + 少量显式上下文构造 **(dataset, sampler, loader, epoch_steps)** ✓
  · 含"单一刻度"校验: ckpt_every 同时定义 epoch 长度/存盘点/评测点 ✓
    （epoch_steps 已降级为**兼容别名**, 配了就必须与 ckpt_every 相等, 否则拒绝启动 ✓）
  · 含 REPA 是否需要原图的判定（repa_need_image: always/never/auto ✓）

⚠ 设计取舍（避免循环导入 ✗）: `aux_dirs` 由**调用方**算好传入 ✓（`aux_dirs_of` 仍在 train.py ✓）;
   同理 world_size / resume_start_step / std_soft_cond / char_remaps 全部显式化 ✓
   （原来是 main() 里的隐式局部变量 ✗）。

与原实现**逐行等价** ✓（只做: 统一减 4 缩进 + 隐式依赖改显式入参 ✓；注释与护栏原样保留 ✓）。
"""
from __future__ import annotations

import logging

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.latent_dataset import MCCDLatentDataset
# ★ 2026-10-13 修正: MCCDDataset 由 **src.utils 包**再导出 ✓（不在 latent_dataset 里 ✗）
#   —— 这是 round 32 我引入的 ImportError ✗（smoke 没抓到, 因为 smoke 不 import train.py ✗）
from src.utils import MCCDDataset
from src.utils.samplers import (DistributedFactorBalancedSampler,
                                LongEpochDistributedSampler)

__all__ = ["build_data"]


def build_data(args, *, rank, world_size, resume_start_step, std_soft_cond,
               char_remaps, aux_dirs, logger=None):
    """构造 (dataset, sampler, loader, epoch_steps) ✓（**逐行等价**于 train.py 原块 ✓）。"""
    logger = logger or logging.getLogger(__name__)
    use_latent = bool(getattr(args, "latent_shards_dir", None))

    if use_latent:
        # ★ 2026-10-11: REPA 是否需要原图 —— **配置项**决定（两机同一份代码, 行为靠配置区分）。
        #   依据: src/loss/losses.py 的 REPALoss 只在特征缓存 miss 的行才用真图;
        #   有完整 repa_cache_dir 时载图纯属浪费 —— 48 上 sym3x(77.8k) 会把 15.3 GB 原图
        #   读进 RAM、每步再 H2D 100 MB, 实测 GPU 利用率被压到 14%。
        #   never 时若真 miss, REPALoss 会**显式报错**（绝不静默失去 REPA）。
        _rnim = str(getattr(args, "repa_need_image", "auto") or "auto").lower()
        # ★ 2026-10-12 修: 原来直接引用 _repa_cache_dir, 而它只在 **REPA 分支内**(w_repa>0) 赋值
        #   -> w_repa=0 时必然 UnboundLocalError（v83 在 48 启动即崩; v77 时代这行还不存在 = 回归）。
        #   改为自足表达式, 不依赖赋值顺序。
        _repa_cache_dir_safe = str(getattr(args, "repa_cache_dir", "") or "")
        if _rnim == "always":
            _repa_need_image = bool(args.w_repa > 0)
        elif _rnim == "never":
            _repa_need_image = False
        else:  # auto
            _repa_need_image = bool(args.w_repa > 0) and (not bool(_repa_cache_dir_safe))
        print(f"[dataset] repa_need_image={_rnim} -> load_image={_repa_need_image} "
              f"(w_repa={args.w_repa}, repa_cache_dir={_repa_cache_dir_safe or '无'})", flush=True)
        dataset = MCCDLatentDataset(csv_file=args.data_csv,
                                    latent_shards_dir=args.latent_shards_dir,
                                    img_root=args.img_root,
                                    image_size=args.image_size,
                                    preload=bool(getattr(args, 'preload', False)),
                                    is_train=std_soft_cond,
                                    load_image=_repa_need_image,
                                    num_preload_workers=int(getattr(args, 'preload_workers', 16)),
                                    char_remap=(char_remaps['char'] if char_remaps else None),
                                    callig_remap=(char_remaps['callig'] if char_remaps else None),
                                    font_remap=(char_remaps['font'] if char_remaps else None),
                                    use_glyph_cond=getattr(args, 'w_glyph_cond', False),
                                    std_soft_root=(getattr(args, 'std_soft_root', '')
                                                   if std_soft_cond else None),
                                    std_soft_sigma_min=(.65 if getattr(args,'std_shape_aug','off')!='off' else float(getattr(
                                        args, 'std_soft_sigma_min', 0.4))),
                                    std_soft_sigma_max=(.65 if getattr(args,'std_shape_aug','off')!='off' else float(getattr(
                                        args, 'std_soft_sigma_max', 1.0))),
                                    std_soft_eval_sigma=float(getattr(
                                        args, 'std_soft_eval_sigma', 0.65)),
                                    skel_latent_shards_dir=(args.skel_latent_shards_dir
                                                            if getattr(args, 'skel_as_glyph_cond', False)
                                                            else None),
                                    # ★ 条件增强: 多个骨架几何变体目录（逗号分隔）。
                                    #   第一个必须是未扰动的原始条件。训练时每样本每步随机选一个。
                                    skel_latent_shards_dirs=(
                                        [d.strip() for d in str(
                                            getattr(args, 'skel_latent_shards_dirs', '') or ''
                                        ).split(',') if d.strip()]
                                        if getattr(args, 'skel_as_glyph_cond', False) else None),
                                    skel_latent_shards_weights=(
                                        getattr(args, 'skel_latent_shards_weights', '') or None
                                        if getattr(args, 'skel_as_glyph_cond', False) else None),
                                    # ⚠ 这里必须把 w_deform_skel 也算进去: 否则开了形变监督
                                    #   但 w_latent_skel/w_std_mid 都为 0 时, batch['inst_skel']
                                    #   根本不会被加载 -> 中间监督静默失效(本项目反复踩的接线缺口)。
                                    inst_skel_shards_dir=(
                                        (getattr(args, 'inst_skel_shards_dir', '') or None)
                                        if (getattr(args, 'w_latent_skel', 0.0) > 0
                                            or getattr(args, 'w_std_mid', 0.0) > 0
                                            or getattr(args, 'w_deform_skel', 0.0) > 0)
                                        else None),
                                    callig_id_map=getattr(args, '_callig_map', None),
                                    callig_script_map=getattr(args, '_callig_script_map', None),
                                    aux_latent_shards_dirs=aux_dirs)
        logger.info("Using latent-cached dataset (skip on-the-fly VAE encode)."
                    + (" preload=ON" if getattr(args, 'preload', False) else ""))
        _aux_dirs = aux_dirs
        if _aux_dirs:
            _aw = getattr(args, 'aux_loss_weights', '') or getattr(args, 'aux_loss_weight', 1.0)
            logger.info(f"[aux] target=12ch+ dirs={_aux_dirs} per-group weights={_aw}")
    else:
        dataset = MCCDDataset(csv_file=args.data_csv, root_dir=args.data_dir, image_size=args.image_size)
    # ── 单一刻度: ckpt_every 同时定义 epoch 长度 / 存盘点 / 评测点 ──────────
    # [2026-09-16] epoch 长度 = **固定步数**（而不是"数据集一遍"）。
    #   起因: 原先 epoch = len(dataset)//batch = 119 步(~21s)，每 119 步 DataLoader
    #   就要重建一次迭代器；而 ckpt + eval 落在 train_steps % ckpt_every 上，
    #   两者互不对齐 -> iterator reset 成为一份**独立**代价，Steps/Sec 出现规律锯齿。
    #
    # ★ 2026-09-17 合并: 原先 `epoch_steps` 与 `ckpt_every` 是两个独立参数，
    #   训练循环里还要用 `min()` 拼出"实际存盘周期"，deferred eval 还得再断言
    #   一次"ckpt 周期整除 eval 周期"。既然**语义上它们本就该相等**（都是"tick"），
    #   干脆合并成一个 —— 于是：
    #     · 存盘条件退化成 `train_steps % _epoch_steps == 0`（一行）
    #     · eval 点 == 存盘点 == epoch 边界，**恒成立**，无需任何断言
    #     · deferred eval 的前置条件自动满足
    #   实测: 121 个配置用 ckpt_every、6 个两者都配且**全部相等**、0 个只配 epoch_steps
    #   -> 保留 ckpt_every 作为唯一参数，**零配置迁移**。
    #   `epoch_steps` 降级为**兼容别名**: 配了就必须与 ckpt_every 相等，否则拒绝启动。
    _epoch_steps = int(getattr(args, 'ckpt_every', 0) or 0)
    _es_alias = int(getattr(args, 'epoch_steps', 0) or 0)
    if _es_alias > 0:
        if _epoch_steps == 0:
            _epoch_steps = _es_alias          # 只配了 epoch_steps 的旧配置
        elif _es_alias != _epoch_steps:
            raise SystemExit(
                f"[epoch] epoch_steps={_es_alias} 与 ckpt_every={_epoch_steps} 不一致。\n"
                f"  两者已合并为**同一个刻度**（epoch 长度 = 存盘周期 = 评测周期），\n"
                f"  不再支持取不同值。请只保留 ckpt_every，或让两者相等。")
    if _epoch_steps < 0:
        _epoch_steps = 0                      # 显式退回旧行为(epoch = 数据集一遍)
    # ckpt_every = 0 -> 不存盘, 且 epoch 退回"数据集一遍"(旧行为)

    # deferred 模式下两处 eval 块整段跳过（见 --eval-mode 的 help）。
    # 注: 不再需要"ckpt 周期整除 eval 周期"的护栏 —— 合并刻度后该条件**恒成立**。
    _EVAL_INLINE = str(getattr(args, 'eval_mode', 'inline')) == "inline"
    if getattr(args,'std_shape_aug','off')!='off':
        if not std_soft_cond or not use_latent:raise ValueError('STD shape requires raw STD + cached image latents')
        from src.utils.std_shape_data import StdShapeDataset
        dataset=StdShapeDataset(dataset,args.std_shape_cache)
    if getattr(args, 'pair_fraction', 0.) > 0:
        if world_size != 1 or resume_start_step:
            raise ValueError('P1 paired sampler requires single-rank fresh optimizer warm-start')
        from src.train.pair_rank import PairMixtureSampler
        sampler = PairMixtureSampler(dataset, int(args.global_batch_size), _epoch_steps,
                                     args.global_seed, args.pair_fraction)
        logger.info('[pair-sampler] paired=%d/%d, script-stratified cells; natural GT endpoints',
                    sampler.n_pair, sampler.batch_size)
    elif args.sampler == "factor_balanced":
        sampler = DistributedFactorBalancedSampler(
            dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
            char_alpha=args.balance_char_alpha,
            callig_alpha=args.balance_callig_alpha)
        logger.info(f"Using factor-balanced sampler: {sampler.summary()} "
                    f"(char_alpha={args.balance_char_alpha}, "
                    f"callig_alpha={args.balance_callig_alpha})")
    else:
        if _epoch_steps > 0:
            sampler = LongEpochDistributedSampler(
                dataset,
                steps_per_epoch=_epoch_steps,
                batch_size=int(args.global_batch_size // world_size),
                num_replicas=world_size,
                rank=rank,
                seed=args.global_seed,
            )
            _samples_per_epoch = (_epoch_steps
                                  * (args.global_batch_size // world_size)
                                  * world_size)
            logger.info(
                f"[sampler] LongEpoch: epoch = {_epoch_steps} 步 "
                f"(= {_samples_per_epoch:,} 样本 = 数据集的 "
                f"{_samples_per_epoch / max(len(dataset), 1):.1f} 倍); "
                f"迭代器每 {_epoch_steps} 步 reset 一次, 与 ckpt_every 对齐。")
            if resume_start_step > 0:
                _inner = sampler.inner_epoch_for_step(resume_start_step)
                sampler.set_start_inner_epoch(_inner)
                logger.info(f"[sampler] resume 数据流对齐: step {resume_start_step} "
                            f"-> 已消耗 {_inner} 个 randperm 份, 从第 {_inner} 份继续")
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.global_seed
            )
    # [INFRA 2026-09-18 实测校正] 曾以为 "preload 下数据已在内存, num_workers 只剩 IPC
    #   开销, 应强制 0" —— **错**。4090 上合成基准 (_review/bench_dataloader_preload.py,
    #   batch=360 / uint8 图像, 复现 preload 返回形态): preload 下 workers 反而更快 ——
    #   num_workers 0→68ms/批, 4→40ms, 8→24ms。原因: __getitem__+default_collate 要对每批
    #   360 样本各做一次 from_numpy 切片 + stack 拷贝 + pin(~71MB/批), workers 把这份
    #   collate **并行化**并与 pin 线程重叠, 收益远超 IPC。故**保留配置的 num_workers**,
    #   不再对 preload 强制 0。
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    logger.info(f"Dataset contains {len(dataset):,} images")
    return dataset, sampler, loader, _epoch_steps, _EVAL_INLINE
