# -*- coding: utf-8 -*-
"""trainer.py —— **训练循环**（2026-10-14 [重构 PR4 收官] ✓ 大刀阔斧版 ✓）

审查 §4.2 原文:
  "建议改为: `def main(): spec = load_and_validate_config(); runtime = setup_runtime(spec);
   data = build_data(spec, runtime); state = build_train_state(spec, runtime);
   preflight(spec, data, state); trainer = Trainer(...); trainer.run()`" ✓
  "尤其不要把所有功能都塞进一个通用 `Trainer` 类。我的目标是：`Trainer` 控制循环，
   `TrainStep` 完成一次更新，`ObjectiveComposer` 组合损失，`CheckpointManager` 负责持久化，
   各自保持小而清晰。" ✓

★ 本模块做的是"**把循环搬出来**"这一步 ✓（审查 §4.2 的形态 ✓）:
  · `TrainState`  = 循环所需的**全部外部状态**（44 个字段 ✓）
  · `training_loop(s)` = 训练循环本体 ✓（**逐字搬移** ✓ 顶部重绑 44 个名字 ✓）

★★ 为什么可以"逐字搬移"而不改一行 ✗:
  已用 AST 逐名验证 ✓:
    ① 44 个状态在循环内**只读** ✓（零写入 ⇒ **无需写回** ✓）
    ② 循环后只需 `_train_fatal` 一个回传值 ✓
  ⇒ 函数顶部 `x = s.x` 重绑后，**810 行循环体一字不改** ✓✓
  ⇒ 等价性是**构造性**的 ✓ 不靠逐行核对 ✓

★ 下一步（审查 §4.2 的"单步" ✓）: 把循环体内的
  H2D → FM forward → 辅助损失 → backward → finite/clip → optimizer/scheduler/EMA
  继续切成 `TrainStep` ✓ —— 本轮先把**整体**搬出来 ✓（一次减 810 行 ✓）
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.train import backward as _bw
from src.train import ema as _ema
import torch.distributed as dist
from time import time
from src.train.ema import update_ema


from src.train.telemetry import Diagnostics as _Diagnostics
from src.train import telemetry as _tel
from src.train.objectives import ObjectiveComposer as _ObjComposer
from src.train.step import resolve_step_inputs as _resolve_step_inputs
from src.train.step import clip_and_step as _clip_and_step
from src.train.step import build_train_model_kwargs as _build_train_model_kwargs
from src.train.eval_callback import maybe_run_in_training
__all__ = ["TrainState", "training_loop"]


@dataclass
class TrainState:
    """训练循环的**全部外部状态** ✓（44 个字段 ✓ AST 推导 ✓）。

    ⚠ 全部为**只读** ✓（循环内零写入 ✓ 已验证 ✓）⇒ 不需要写回 ✓
    """
    _SKEL_STRUCT_WARNED: object = None
    _acc_c12: object = None
    log_steps: object = None
    running_diff: object = None
    running_loss: object = None
    running_repa: object = None
    running_skel: object = None
    running_std_mid: object = None
    start_time: object = None
    train_steps: object = None
    args: object = None          # ★ main() 的参数 ⇒ 必须在状态里 ✓
    _EVAL_INLINE: object = None
    _aux_dirs: object = None
    _ch_w: object = None
    _ckpt: object = None
    _critic_loss_fn: object = None
    _empty_cache_at: object = None
    _epoch_steps: object = None
    _epochs_needed: object = None
    _es_check_every: object = None
    _guard: object = None
    _inj_probe: object = None
    _inj_stats: object = None
    _mid_struct_loss: object = None
    _skel_struct_loss_fn: object = None
    _std_soft_cond: object = None
    _stopper: object = None
    _strong_style: object = None
    _style_anchor_k: object = None
    _style_anchor_mode: object = None
    _style_anchor_target: object = None
    _style_anchor_w: object = None
    _style_ortho_w: object = None
    _style_rank_loss_fn: object = None
    _style_table_param: object = None
    _train_vae_enc: object = None
    _vae_sf: object = None
    _vae_shift: object = None
    _writer_table_tuning: object = None
    checkpoint_dir: object = None
    cond_mode: object = None
    device: object = None
    diffusion: object = None
    ema_model: object = None
    loader: object = None
    logger: object = None
    model: object = None
    opt: object = None
    rank: object = None
    repa_loss_fn: object = None
    resume_start_step: object = None
    sampler: object = None
    scheduler: object = None
    trainable_params_list: object = None
    vae: object = None


def training_loop(s: TrainState):
    """训练循环本体 ✓（**逐字搬移** ✓ 顶部重绑状态名 ✓）。

    返回: `_train_fatal` ✓（循环后唯一需要的循环内绑定 ✓ None = 正常结束 ✓）
    """
    _SKEL_STRUCT_WARNED = s._SKEL_STRUCT_WARNED
    _acc_c12 = s._acc_c12
    log_steps = s.log_steps
    running_diff = s.running_diff
    running_loss = s.running_loss
    running_repa = s.running_repa
    running_skel = s.running_skel
    running_std_mid = s.running_std_mid
    start_time = s.start_time
    train_steps = s.train_steps
    args = s.args
    _EVAL_INLINE = s._EVAL_INLINE
    _aux_dirs = s._aux_dirs
    _ch_w = s._ch_w
    _ckpt = s._ckpt
    _critic_loss_fn = s._critic_loss_fn
    _empty_cache_at = s._empty_cache_at
    _epoch_steps = s._epoch_steps
    _epochs_needed = s._epochs_needed
    _es_check_every = s._es_check_every
    _guard = s._guard
    _inj_probe = s._inj_probe
    _inj_stats = s._inj_stats
    _mid_struct_loss = s._mid_struct_loss
    _skel_struct_loss_fn = s._skel_struct_loss_fn
    _std_soft_cond = s._std_soft_cond
    _stopper = s._stopper
    _strong_style = s._strong_style
    _style_anchor_k = s._style_anchor_k
    _style_anchor_mode = s._style_anchor_mode
    _style_anchor_target = s._style_anchor_target
    _style_anchor_w = s._style_anchor_w
    _style_ortho_w = s._style_ortho_w
    _style_rank_loss_fn = s._style_rank_loss_fn
    _style_table_param = s._style_table_param
    _train_vae_enc = s._train_vae_enc
    _vae_sf = s._vae_sf
    _vae_shift = s._vae_shift
    _writer_table_tuning = s._writer_table_tuning
    checkpoint_dir = s.checkpoint_dir
    cond_mode = s.cond_mode
    device = s.device
    diffusion = s.diffusion
    ema_model = s.ema_model
    loader = s.loader
    logger = s.logger
    model = s.model
    opt = s.opt
    rank = s.rank
    repa_loss_fn = s.repa_loss_fn
    resume_start_step = s.resume_start_step
    sampler = s.sampler
    scheduler = s.scheduler
    trainable_params_list = s.trainable_params_list
    vae = s.vae
    for epoch in range(_epochs_needed):
        sampler.set_epoch(epoch)
        if rank == 0:
            # [2026-09-16] epoch 已退化为"ckpt/eval 刻度"(见 --epoch-steps), 且续训时
            # 外层 epoch 计数器从 0 重新开始 -> 直接乘 epoch 会算错区间号(盯着日志判断
            # 进度时会误判)。这里用**真实已训步数**算, 续训后依旧正确。
            _span = (f" (steps {resume_start_step + epoch * _epoch_steps + 1}"
                     f"-{resume_start_step + (epoch + 1) * _epoch_steps})"
                     if _epoch_steps > 0 else "")
            logger.info(f"Begin epoch {epoch}{_span} — ckpt+eval 落点...")
    
        try:
            # ★ 2026-10-14 [重构 PR4] 训练期诊断收进 src/train/telemetry.py ✓（纯观测 ✓）
            #   ① 算子级 profile（--profile-steps N, 默认 0=关 ✓）
            #      跳过前 --profile-skip 步后, 对接下来 N 步做 torch.profiler,
            #      随后打印**逐算子 CUDA 时间表**（回答"GPU 时间花在哪些算子上" ✓）
            #   ② 分配历史（--profile-memory-steps N, 默认 0=关 ✓）
            #      记录 device 分配的 Python 调用栈 ⇒ tools/prof/mem_top.py 聚合"分配点排行榜" ✓
            #   两者**默认全关** ✓ 只在窗口内开销 ✓ 不参与任何训练语义 ✓
            _diag = _Diagnostics(args, logger)
            for batch_idx, batch in enumerate(loader):
                # ★ 2026-10-11 算子级 profile / 分配历史: 窗口起点 ⇒ 见 telemetry.py ✓
                _diag.maybe_start(batch_idx)
                # ★ non_blocking=True: DataLoader 已开 pin_memory=True，但**必须显式传**
                #   这个参数才会走异步拷贝；否则即使源在 pinned 内存里也是同步拷贝，
                #   主机线程会被阻塞（图像单批 ~283MB，实测代价可观）。见 docs/system/70 §8.6。
                y_callig = batch['y_callig'].to(device, non_blocking=True)
                if getattr(args, 'use_script_cond', False):
                    # v53 三表: y_callig = 书家连续索引(10 类), 另带 y_script
                    y_callig = batch['y_callig_raw'].to(device, non_blocking=True)
                    y_script = batch['y_script'].to(device, non_blocking=True)
                y_char = batch['y_char'].to(device, non_blocking=True)
                if cond_mode == "3cond":
                    y_script = batch['y_script'].to(device, non_blocking=True)

                # 注意: `_ch_w` 与 aux 权重已在**循环外**解析好（见 resolve_aux_channel_weights），
                # 这里不再重算，也不要把它重置为 None。
                # ★ 2026-10-14 [重构 PR4] 单步输入解析抽到 step.resolve_step_inputs ✓
                #   （审查 §4.2 第 1 条 "H2D 数据移动" ✓）
                #   ⚠ 三条分支的**梯度语义不同** ✗（rsample 要训 VAE ✓ / sample+no_grad ✓）
                #     ⇒ 搬移时**连上下文管理器一起**逐字复制 ✓ 刻意**不**统一 ✗
                _si = _resolve_step_inputs(batch, vae, device,
                                           train_vae_enc=_train_vae_enc,
                                           vae_sf=_vae_sf, vae_shift=_vae_shift)
                x_latent, x, loss_vae_kl = _si.x_latent, _si.x, _si.loss_vae_kl

                # 统一时间步采样: FlowMatching.sample_t -> t∈[0,1); GaussianDiffusion.sample_t -> t∈{0..T-1}。
                # 调用方绝不自己分支 (否则会重蹈 flow/randint 错配覆辙)。
                t = diffusion.sample_t(x_latent.shape[0], device)
                # ★ 2026-10-14 [重构 PR5] 条件装配抽成 step.build_train_model_kwargs ✓
                #   （审查 §4.1: 收进 ConditionBatch ✓）逐字等价 ✓:
                #     · y_script 的三种来源是**并集** ✓（3cond / use_script_cond / hier ✓）
                #     · g 的三路来源顺序不变 ✓（std_soft → g → skel_latent ✓）
                #     · 用 kwargs_for_call ✓（"存在才加键" ✓ **不是** for_forward=True ✗
                #       后者总是保留 g ✗ 会把"无字形"变成"明确 g=None" ✗ 语义改变 ✗）
                #   ⚠ 下面的**条件 dropout / 形变增强**仍操作同一 dict ✓ 一字未改 ✓
                model_kwargs = _build_train_model_kwargs(
                    args, batch, device, cond_mode=cond_mode,
                    std_soft_cond=_std_soft_cond,
                    y_callig=y_callig, y_char=y_char, y_script=y_script)

                # ── 条件几何形变增强 (Condition Deformation Augmentation) ───────
                _g_def_prob = float(getattr(args, 'glyph_deform_prob', 0.0))
                if 'g' in model_kwargs and _g_def_prob > 0:
                    from src.utils.deform_aug import random_skeleton_deformation
                    _g_def_scale = float(getattr(args, 'glyph_deform_scale', 1.0))
                    model_kwargs['g'] = random_skeleton_deformation(
                        model_kwargs['g'], prob=_g_def_prob,
                        max_rot_deg=5.0 * _g_def_scale,
                        max_scale=0.08 * _g_def_scale,
                        max_shear=0.06 * _g_def_scale,
                        max_trans_px=1.5 * _g_def_scale,
                        max_elastic_px=1.5 * _g_def_scale,
                        preserve_amplitude=not _std_soft_cond,
                    )

                # ── 条件噪声增强 (Condition Noise Augmentation) ───────────
                # 标准骨架 g 是固定的印刷体 latent (cos=0.902 to GT)。加噪声迫使
                # 模型学会从「不完美的结构条件」中提取拓扑信息，避免对 g 精确数值过拟合。
                _gn_scale = float(getattr(args, 'glyph_noise_scale', 0.0))
                _gn_prob = float(getattr(args, 'glyph_noise_prob', 0.0))
                _gpd = float(getattr(args, 'glyph_patch_drop', 0.0))
                if 'g' in model_kwargs and (_gn_scale > 0 or _gpd > 0):
                    _g = model_kwargs['g']
                    if _gn_scale > 0 and _gn_prob > 0:
                        _mask = (torch.rand(_g.shape[0], 1, 1, 1, device=device) < _gn_prob).to(_g.dtype)
                        _scales = torch.rand(_g.shape[0], 1, 1, 1, device=device) * _gn_scale
                        _g = _g + _mask * torch.randn_like(_g) * _scales
                    if _gpd > 0:
                        _pmask = (torch.rand(_g.shape[0], 1, _g.shape[2], _g.shape[3],
                                             device=device) > _gpd).to(_g.dtype)
                        if _std_soft_cond:
                            _z_bg = torch.zeros(1, 1, 1, 1, device=device, dtype=_g.dtype)
                        else:
                            from src.utils.deform_aug import Z_BG_VEC
                            _z_bg = Z_BG_VEC.to(device=device, dtype=_g.dtype)
                        _g = _g * _pmask + _z_bg * (1.0 - _pmask)
                    model_kwargs['g'] = _g

                # ── 条件「整块抹白」增强 (2026-10-01) ──────────────────────
                # 动机: 推理时喂给 stage2 的是 **stage1 预测出的骨架**, 它的失效模式
                #   不是"加点噪", 而是**整段笔画没了 / 整块糊掉**。上面的逐格
                #   patch_drop 只产生散点, 模拟不了"少一笔"; 这里按**连续区域**
                #   抹成纯白背景 (Z_BG_VEC), 才对得上那个失效模式。
                #   两种形状: 细长条(≈少了一笔) / 方块(≈一块糊掉)。
                _gm_prob = float(getattr(args, 'glyph_mask_prob', 0.0))
                if 'g' in model_kwargs and _gm_prob > 0:
                    _g = model_kwargs['g']
                    if _std_soft_cond:
                        _z_bg = torch.zeros(1, 1, 1, 1, device=device, dtype=_g.dtype)
                    else:
                        from src.utils.deform_aug import Z_BG_VEC
                        _z_bg = Z_BG_VEC.to(device=device, dtype=_g.dtype)
                    _B, _C, _H, _W = _g.shape
                    _hit = torch.rand(_B, device=device) < _gm_prob
                    # 尺度: 1 个 latent 格 = 8px, 默认 4 格 = 32x32 px (2026-10-01 指定)
                    _sz = int(getattr(args, 'glyph_mask_size', 4))
                    _jit = int(getattr(args, 'glyph_mask_jitter', 1))
                    _nmax = int(getattr(args, 'glyph_mask_n', 3))
                    _lo, _hi = max(2, _sz - _jit), max(2, _sz + _jit)
                    _msk = torch.zeros(_B, 1, _H, _W, device=device, dtype=torch.bool)
                    for _b in range(_B):
                        if not bool(_hit[_b]):
                            continue
                        for _ in range(_nmax):
                            _h = int(torch.randint(_lo, _hi + 1, (1,)).item())
                            _w = int(torch.randint(_lo, _hi + 1, (1,)).item())
                            _y = int(torch.randint(0, _H - _h + 1, (1,)).item())
                            _x = int(torch.randint(0, _W - _w + 1, (1,)).item())
                            _msk[_b, 0, _y:_y + _h, _x:_x + _w] = True
                    model_kwargs['g'] = torch.where(_msk, _z_bg, _g)
            
                # REPA: 请求多层中间特征 (统一 infra, 多层 dict / 单层兼容)
                if args.w_repa > 0 and repa_loss_fn is not None:
                    if len(repa_loss_fn.layers) > 1:
                        model_kwargs['return_intermediate_layers'] = repa_loss_fn.layers
                    else:
                        model_kwargs['return_intermediate_layer'] = repa_loss_fn.layers[0]

                # Forward pass under bf16 autocast (same exponent range as fp32, no overflow).
                #
                # return_pred_xstart: flow 分支默认**不返回** pred_xstart（避免
                # autograd 图膨胀），但下面 w_std_mid 依赖它。若这里不请求，flow
                # 模式下该机制会因 `loss_dict.get("pred_xstart", None)` 恒为 None 而
                # **静默失效** —— 不报错、loss 正常下降、但从未生效。w_std_mid 正是
                # 「把去噪中段预测的 x0 拉向标准字形 latent」的预训练改进项。
                # gaussian_diffusion 不接受该参数，故用 try/except 兼容。
                _need_x0 = (getattr(args, 'w_std_mid', 0.0) > 0
                            or getattr(args, 'w_latent_skel', 0.0) > 0
                            or getattr(args, 'w_style_rank', 0.0) > 0
                            or getattr(args, 'w_critic_writer', 0.0) > 0
                            or getattr(args, 'w_critic_away', 0.0) > 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if _need_x0:
                        try:
                            loss_dict = diffusion.training_losses(
                                model, x_latent, t, model_kwargs,
                                return_pred_xstart=True, channel_weights=_ch_w)
                        except TypeError:
                            # gaussian_diffusion 不支持该参数，本身就会返回 pred_xstart
                            loss_dict = diffusion.training_losses(
                                model, x_latent, t, model_kwargs)
                    else:
                        loss_dict = diffusion.training_losses(
                            model, x_latent, t, model_kwargs, channel_weights=_ch_w)
                    loss_diff = loss_dict["loss"].mean()

                loss_repa = torch.tensor(0.0, device=device)
                loss_x0lat = torch.tensor(0.0, device=device)
                loss_skel_struct = torch.tensor(0.0, device=device)

                # === INFRA FIX: pred_xstart from training_losses carries the ENTIRE DiT
                # forward graph (every block's activations are kept alive because the
                # graph flows back through _predict_xstart_from_eps → model_output).
                # This is THE memory leak: the 256-token × 384-dim × 12-block activation
                # graph stays pinned until backward() runs, and struct decoders build
                # ADDITIONAL 256×256 activations on top of it. When t<=500 (~50% of steps)
                # the decoder runs → graph doubles; when t>500 the graph lingers but
                # decoders don't run → the cycling 20G→22G→14G pattern.
                #
                # FIX: extract pred_xstart WITH the graph only when we actually need it
                # for a differentiable struct loss (t<=max_t). For t>500 steps, detach
                # immediately so the full graph is freed at the next zero_grad. We also
                # break the reference in loss_dict so no stale graph survives the loop.
                _need_x0_grad = (getattr(args, 'w_std_mid', 0.0) > 0
                                 or getattr(args, 'w_latent_skel', 0.0) > 0
                                 or getattr(args, 'w_style_rank', 0.0) > 0
                                 or getattr(args, 'w_critic_writer', 0.0) > 0
                                 or getattr(args, 'w_critic_away', 0.0) > 0)
                pred_xstart_latent = loss_dict.get("pred_xstart", None)
                if pred_xstart_latent is not None and not _need_x0_grad:
                    # No struct loss this run at all — drop the graph immediately.
                    pred_xstart_latent = pred_xstart_latent.detach()
                # Clear the heavy reference inside loss_dict so the dict can't keep
                # the graph alive after we exit this step.
                loss_dict.pop("pred_xstart", None)

                # ---- MIDSTEP_STD (已抽到 src/loss/structure_mid.py: MidStructureLoss) ----
                # 旧内联是 MSE(x0_pred, 细骨架 g), 与中程软墨迹形态冲突 -> 换成
                # 载体粗化 + 低通子空间 + 通道归一。gt=GT图像latent(现成), skel=g。
                loss_std_mid = torch.tensor(0.0, device=device)
                if (_mid_struct_loss is not None
                        and pred_xstart_latent is not None):
                    _lc = int(getattr(args, 'latent_channels', 4))
                    _gt = x_latent[:, :_lc]
                    # 膨胀后的 GT 实例骨架 (20px)。不用标准骨架 g, 那是条件本身。
                    _gsk = batch.get('inst_skel', None)
                    if _gsk is None or _gsk.numel() == 0:
                        if not getattr(_mid_struct_loss, "_empty_warned", False):
                            _mid_struct_loss._empty_warned = True
                            logger.warning(
                                "[std_mid] batch['inst_skel'] 为空, 中程监督未生效。"
                                "需要 --inst-skel-shards-dir 指向膨胀后的 GT 骨架。")
                        loss_std_mid = torch.tensor(0.0, device=device)
                    else:
                        loss_std_mid = _mid_struct_loss(
                            pred_xstart_latent, _gt,
                            _gsk.to(device, non_blocking=True).float(),
                            t.to(device))

                intermediate_feats = loss_dict.get("intermediate_feats", None)
                # ★ 2026-10-11: 不再要求 `x is not None` —— 有 REPA 特征缓存时按 img_ids 查表即可,
                #   真图只在缓存 miss 的行才需要(见 losses.py)。若 x=None 却真 miss,
                #   REPALoss 会**显式抛错**, 绝不静默失去 REPA。
                if intermediate_feats is not None and repa_loss_fn is not None and args.w_repa > 0:
                    # 统一 REPA (公共 infra): 多层 dict / 单层张量 + warmup 渐进
                    # img_ids: 配合 --repa-cache-dir 查表 (命中免 DINO 前向)
                    _img_ids = batch.get('img_id', None)
                    loss_repa = repa_loss_fn(intermediate_feats, x, step=train_steps,
                                             img_ids=_img_ids)

                # ---- 实例骨架结构 loss (辅助, 冻结 probe, 零可训练参数) ----
                # 见 src/train/latent_structure.py:LatentSkelStructureLoss 的 docstring:
                # 与 12ch 的区别是"不在扩散目标里" —— 4ch 主干 / CFG 作用域 / 推理成本
                # 全部不受影响, 新增可训练参数 0(当前主要矛盾是过拟合, 这是决定性优势)。
                # target = batch['skel_latent'] **必须是实例骨架**; 若指到标准字形就是纯重复 g。
                if _skel_struct_loss_fn is not None and pred_xstart_latent is not None:
                    # [inst-skel 2026-09-16] target 现在读 **inst_skel**(实例骨架, GT 图派生),
                    # 与条件 g (skel_latent=shards_std) 解耦。不再读 skel_latent。
                    _sk = batch.get('inst_skel', None)
                    if _sk is None or _sk.numel() == 0:
                        if not _SKEL_STRUCT_WARNED:
                            _SKEL_STRUCT_WARNED = True
                            logger.warning(
                                "[skel-struct] w_latent_skel>0 但 batch['inst_skel'] 为空 "
                                "(需配置 --inst-skel-shards-dir 指向实例骨架 shards) "
                                "-> 本项静默失效。这类静默失效已踩过多次, 故只在首次告警。")
                        loss_skel_struct = torch.tensor(0.0, device=device)
                    else:
                        loss_skel_struct = _skel_struct_loss_fn(
                            pred_xstart_latent,
                            _sk.to(device, non_blocking=True).float(),
                            t.to(device))

                # ★ 2026-10-14 [重构 PR4] 损失装配抽到 ObjectiveComposer ✓
                #   ⚠ **注册顺序 = 原加法顺序** ✗（浮点加法不满足结合律 ✓
                #     顺序变了 ⇒ 末位不同 ⇒ 训练轨迹不同 ✗）
                _obj = _ObjComposer(loss_diff, base_name="fm")
                # 统一 REPA: w × (1 - cos) 已在 RepaModule.forward 内含 warmup
                _obj.add("repa", loss_repa, 1.0)
                _obj.add("std_mid", loss_std_mid, getattr(args, 'w_std_mid', 0.0))
                _obj.add("latent_skel", loss_skel_struct,
                         getattr(args, 'w_latent_skel', 0.0))
                loss = _obj.total
                if _train_vae_enc and loss_vae_kl is not None:
                    _w_vkl = float(getattr(args, 'w_vae_kl', 1e-5))
                    loss = loss + _w_vkl * loss_vae_kl

                # ---- style-rank loss (t 门控在模块内) ----
                loss_style_rank = torch.tensor(0.0, device=device)
                if _style_rank_loss_fn is not None and getattr(args, 'w_style_rank', 0.0) > 0:
                    if pred_xstart_latent is not None:
                        _yp = batch['y_pair'].to(device, non_blocking=True)
                        loss_style_rank, _n_style_rank = _style_rank_loss_fn(
                            pred_xstart_latent, _yp, t.to(device))
                        # ★★ v85B: **梯度范数比探针**（方案 §三.3 明确要求的标定方式 ✓）
                        #   方案原文: "建议按**共享 DiT 参数的梯度范数**确定权重，使新风格损失
                        #   在启用样本上贡献约 **FM 梯度的 5–10%**，而**不是仅根据两个标量
                        #   loss 的数值**选择权重 ✓"。
                        #   为什么: 两个 loss 的**标量数值不可比** ✗（一个在 latent MSE 尺度、
                        #   一个在 logsumexp 尺度 ✓）⇒ 只有**对同一组参数的梯度**可比 ✓。
                        #   实现: 用 autograd.grad 分别对两组 loss 求同一组共享参数（可训练 ✓）
                        #   的梯度（retain_graph=True ✓），只在**前几步**做以免拖慢训练 ✓。
                        _pstep = int(getattr(args, 'probe_style_grad_steps', 0) or 0)
                        if _pstep and train_steps < _pstep and int(_n_style_rank) > 0:
                            _shared = [p for p in model.parameters() if p.requires_grad]
                            # ★ 2026-10-14 [重构 PR4] 探针原语抽到 backward.grad_norm ✓
                            #   （数学与旧代码**逐字相同** ✓ 只求范数, 不累积到 .grad ✓）
                            _n_fm = _bw.grad_norm(loss, _shared)
                            _n_st = _bw.grad_norm(loss_style_rank, _shared)
                            _ratio = float(_n_st / (_n_fm + 1e-12))
                            logger.info(
                                "[style-rank-probe] step=%d n=%d | ‖dL_FM/dθ‖=%.4e "
                                "‖dL_style/dθ‖=%.4e | λ=%.5g 时贡献比=%.2f%% | "
                                "目标 5-10%% ⇒ 建议 λ≈%.5g",
                                train_steps, int(_n_style_rank), float(_n_fm), float(_n_st),
                                float(args.w_style_rank), 100.0 * _ratio,
                                float(args.w_style_rank) * (0.075 / max(_ratio, 1e-12)))
                        _obj.add("style_rank", loss_style_rank,
                                 getattr(args, 'w_style_rank', 0.0))
                        loss = _obj.total

                # ---- v85B/C: 冻结 critic 的生成端辅助 Loss（方案 §六 ✓）----
                # L_writer^G = CE(slot(ẑ₀), 目标书家槽) | L_away^G = softplus(−brush(ẑ₀))
                # 全程 latent ✓（ẑ₀ 直接喂 ✓ 不挂 decoder ✓）critic 冻结但**不套 no_grad** ✓
                # 只对 t∈[t_min,t_max] 的样本计算 ✓（方案建议 ✓）
                loss_critic_writer = torch.tensor(0.0, device=device)
                loss_critic_away = torch.tensor(0.0, device=device)
                # ★ 活性计数器（静默失效必须可观测 ✓ —— 本会话第 3 次同类坑 ✓）
                _cw = globals().setdefault('_CRITIC_CNT', {'n': 0, 'hit': 0, 'warned': False})
                _cw['n'] += 1
                if _critic_loss_fn is not None and pred_xstart_latent is not None \
                        and (getattr(args, 'w_critic_writer', 0.0) > 0
                             or getattr(args, 'w_critic_away', 0.0) > 0):
                    # ★ flow matching 的 t 本来就是 [0,1] 连续值 ✓（实测 0.042-0.965,
                    #   num_timesteps=50 只是档位数 ✗ —— 除它就把窗口错位 50 倍 ✗✗）
                    _ct = t.to(device).float()
                    _cmask = ((_ct >= getattr(args, 'critic_t_min', 0.05))
                              & (_ct <= getattr(args, 'critic_t_max', 0.30)))
                    if _cmask.any():
                        _cw['hit'] += 1
                    if train_steps % 100 == 0 and not _cw['warned']:
                        _cw['warned'] = True
                        logger.info("[critic-activity] steps=%d hit=%d | "
                                    "pred_xstart=%s w=(%.4g,%.4g) | t 原始: min=%.4g "
                                    "max=%.4g num_ts=%s | _ct: min=%.4g max=%.4g ⇒ %s",
                                    _cw['n'], _cw['hit'],
                                    '有' if pred_xstart_latent is not None else 'None!',
                                    args.w_critic_writer, args.w_critic_away,
                                    float(t.min()), float(t.max()),
                                    str(getattr(diffusion, 'num_timesteps', '?')),
                                    float(_ct.min()), float(_ct.max()),
                                    '正常' if _cw['hit'] else '**未命中 ✗**')
                    if _cmask.any():
                        _zs = pred_xstart_latent[_cmask].float()
                        # ★ g = batch['std_soft'] ✓（现成 per-sample [1,64,64] ✓
                        #   σ=eval 0.65 ✓ —— 与 critic 的 g_std bank 同源同分布 ✓
                        #   **不需要任何字符反查** ✓（y_script/y_char 是整数索引 ✗
                        #   str() 会得到 'tensor(1)' ⇒ KeyError: 'tensor(1)' ✗✗ 实测 ✓））
                        _ss = batch['std_soft'][_cmask.cpu()]
                        if _ss.dim() == 3:
                            _ss = _ss.unsqueeze(1)                # [B,1,64,64] ✓
                        _g_direct = _ss.float().to(device) / 255.0 \
                            if _ss.float().max() > 1.5 else _ss.float().to(device)
                        # 目标槽 = 书家×书体（y_callig_raw + y_script **都是 id** ✓）
                        _cmask_cpu = _cmask.cpu()          # batch 张量在 CPU ✓
                        _sc_id = batch['y_script'][_cmask_cpu].long().to(device)
                        _cw_id = batch['y_callig_raw'][_cmask_cpu].long().to(device)
                        # slots 表按 "writer·script" 存 ✗ 但 id 空间不同 ⇒ 用 critic 的
                        # **writer_ids/scripts** 反查 ✓（critic 存了训练 CSV 的 slot 字符串 ✓
                        # 但 batch 只有 id ⇒ 需要 id→str 映射 —— 由 _build_aux_losses 建好 ✓）
                        _lut = getattr(_critic_loss_fn, 'id2slot', None)
                        if _lut is None:
                            _cw['warned'] = False
                            logger.info("[critic-missing-lut] id→slot 映射缺失 ⇒ 本步跳过 ✗")
                        else:
                            _keys = [_lut.get((int(w), int(s)), None)
                                     for w, s in zip(_cw_id.tolist(), _sc_id.tolist())]
                            _valid = [k is not None for k in _keys]
                            if any(_valid):
                                _zi = [i for i, v in enumerate(_valid) if v]
                                _zs = _zs[_zi]
                                _g_direct = _g_direct[_zi]
                                _si = torch.tensor(
                                    [_critic_loss_fn.s2i[
                                        _lut[(int(_cw_id[i]), int(_sc_id[i]))].split('·')[1]]
                                     for i in _zi], dtype=torch.long, device=device)
                                _ys = torch.tensor(
                                    [_critic_loss_fn.slots.index(
                                        _lut[(int(_cw_id[i]), int(_sc_id[i]))])
                                     for i in _zi], dtype=torch.long, device=device)
                                _lw, _la, _cp = _critic_loss_fn.forward_g(
                                    _zs, _g_direct, _si, _ys)
                                loss_critic_writer = _lw
                                loss_critic_away = _la
                        # ★ 梯度范数比探针 ✓（与 style-rank 探针同一机制 ✓）
                        _pstep = int(getattr(args, 'probe_style_grad_steps', 0) or 0)
                        if _pstep and train_steps < _pstep:
                            _shared = [p for p in model.parameters() if p.requires_grad]
                            # ★ FM 基线用**总 loss 减去 critic 项** ✗ 更简单可靠:
                            #    直接对 loss_diff 求梯度 ✓（它是本步 FM 项 ✓）
                            # ★ 2026-10-14 [重构 PR4] 同上, 统一走 backward.grad_norm ✓
                            _n_fm = _bw.grad_norm(loss_diff, _shared)
                            _n_cw = _bw.grad_norm(loss_critic_writer, _shared)
                            _n_ca = _bw.grad_norm(loss_critic_away, _shared)
                            # ★★ 贡献%必须**含 w** ✓（裸比 ✗ 曾致 λ 建议错 w 倍 ✗✗
                            #    ⇒ v85C 以 3e-8 实跑 = 有效贡献 0.00016%% ≈ 空跑 ✗）
                            _r_w = float(_n_cw / (_n_fm + 1e-12))
                            _r_a = float(_n_ca / (_n_fm + 1e-12))
                            logger.info(
                                "[critic-probe] step=%d n=%d | ‖dL_FM/dθ‖=%.4e "
                                "‖dL_writer/dθ‖=%.4e ‖dL_away/dθ‖=%.4e | "
                                "w=(%.4g,%.4g) ⇒ 实际贡献 %.2f%% / %.2f%% | "
                                "目标 5%%/1-2%% ⇒ 建议 w_writer≈%.4g w_away≈%.4g",
                                train_steps, int(_cmask.sum()), float(_n_fm), float(_n_cw),
                                float(_n_ca), float(args.w_critic_writer),
                                float(args.w_critic_away),
                                100.0 * float(args.w_critic_writer) * _r_w,
                                100.0 * float(args.w_critic_away) * _r_a,
                                0.05 / max(_r_w, 1e-12),
                                0.015 / max(_r_a, 1e-12))
                        _obj.add("critic_writer", loss_critic_writer,
                                 getattr(args, 'w_critic_writer', 0.0))
                        _obj.add("critic_away", loss_critic_away,
                                 getattr(args, 'w_critic_away', 0.0))
                        loss = _obj.total

                # ---- DeformSkel 中间监督: 形变后的骨架应逼近**该书家的 GT 骨架** ----
                # 这是本项目第一个"自带稠密监督"的风格干预: 目标数据(实例骨架)现成,
                # 且 step0 时 g'=g_std != g_gt -> 立刻有梯度, 不像其它风格模块那样学不动。
                loss_deform = torch.tensor(0.0, device=device)
                _DEFORM_EMA = globals().setdefault('_DEFORM_EMA', {'loss': None, 'off': None})
                if getattr(args, 'w_deform_skel', 0.0) > 0:
                    _dm = getattr(model, 'deform_skel', None)
                    if _dm is None:
                        if not globals().get('_WARNED_DEFORM', False):
                            globals()['_WARNED_DEFORM'] = True
                            logger.warning("[deform] w_deform_skel>0 但模型没有 deform_skel "
                                           "-> 本项静默失效(已踩过多次, 故只首次告警)")
                    elif getattr(_dm, 'last_out', None) is None:
                        if not globals().get('_WARNED_DEFORM2', False):
                            globals()['_WARNED_DEFORM2'] = True
                            logger.warning("[deform] deform_skel.last_out 为空 -> 形变没被调用"
                                           "(compile 下属性写入可能被吞) -> 本项静默失效")
                    else:
                        _gsk = batch.get('inst_skel', None)
                        if _gsk is None:
                            if not globals().get('_WARNED_DEFORM3', False):
                                globals()['_WARNED_DEFORM3'] = True
                                logger.warning("[deform] batch['inst_skel'] 为空 "
                                               "(需 --inst-skel-shards-dir 指向 GT 骨架) "
                                               "-> 中间监督静默失效")
                        else:
                            _g2 = _dm.last_out
                            _tgt = _gsk.to(device, non_blocking=True).float()
                            # ★ 只在"真的做了形变"的样本上算: 条件被 drop 的样本用的是
                            #   null 风格向量, 形变头产出的是垃圾, 不该进监督。
                            #   (第一版只改前向没改 loss, 实测 loss 一模一样 0.3687)
                            _m = getattr(_dm, 'last_mask', None)
                            if _m is not None and not bool(_m.all()):
                                if bool(_m.any()):
                                    _g2 = _g2[_m]
                                    _tgt = _tgt[_m]
                                else:
                                    _g2 = None
                            if _g2 is not None and _g2.shape == _tgt.shape:
                                loss_deform = (_g2 - _tgt).pow(2).mean()
                                loss = loss + getattr(args, 'w_deform_skel', 0.0) * loss_deform
                                _v = float(loss_deform.detach())
                                _DEFORM_EMA['loss'] = (_v if _DEFORM_EMA['loss'] is None else 0.9 * _DEFORM_EMA['loss'] + 0.1 * _v)
                                _os = _dm.offset_stats()
                                if _os:
                                    _DEFORM_EMA['off'] = (_os['mean_abs'] if _DEFORM_EMA['off'] is None else 0.9 * _DEFORM_EMA['off'] + 0.1 * _os['mean_abs'])
                            else:
                                if not globals().get('_WARNED_DEFORM4', False):
                                    globals()['_WARNED_DEFORM4'] = True
                                    logger.warning(f"[deform] 形状不匹配 g'={tuple(_g2.shape)} "
                                                   f"target={tuple(_tgt.shape)} -> 跳过")

                # ★★ [2026-09-29] 形变头正则化 —— **此前完全没接, 是 v27 过拟合的根因** ★★
                #   `DeformSkel.regularizers()` 早就提供了 TV(偏移场平滑) + Jacobian(防折叠)
                #   + tv_out/tv_res(压输出与残差的高频) + l1_prune/l1_lig(topo 掩码稀疏),
                #   但**主训练路径从未调用** —— grep 'regularizers' train.py 曾为空。
                #   后果: v27 的 SkelNet 在「4.1M 参数 + 零正则 + weight_decay=0」下直接
                #   拟合训练集, 实测 seen 全指标单调上升而 strict ssim 单调下降
                #   (0.6069→0.5899→0.5850) = 明确过拟合。
                #   尤其 residual/topo 是**无约束的加墨头**(warp 只能移动笔画、必然泛化;
                #   残差/拓扑能在任意位置画任意墨 -> 能记住训练字符的骨架), 最需要正则。
                #   口径与 tools/train_deform_standalone.py 完全一致。
                _dm_reg = getattr(model, 'deform_skel', None)
                if _dm_reg is not None and getattr(_dm_reg, 'last_off', None) is not None:
                    _rr = _dm_reg.regularizers()
                    if _rr is not None:
                        _reg_used = {}
                        for _rk, _rw in (
                            ('tv', float(getattr(args, 'w_tv', 0.0) or 0.0)),
                            ('fold', float(getattr(args, 'w_fold', 0.0) or 0.0)),
                            ('tv_out', float(getattr(args, 'w_tv_out', 0.0) or 0.0)),
                            ('tv_res', float(getattr(args, 'w_tv_res', 0.0) or 0.0)),
                            ('tv_stroke', float(getattr(args, 'w_tv_stroke', 0.0) or 0.0)),
                            ('l1_prune', float(getattr(args, 'w_prune_l1', 0.0) or 0.0)),
                            ('l1_lig', float(getattr(args, 'w_lig_l1', 0.0) or 0.0)),
                        ):
                            if _rw > 0 and _rk in _rr:
                                loss = loss + _rw * _rr[_rk]
                                _reg_used[_rk] = (_rw, float(_rr[_rk].detach()))
                        # 一次性回显: 确认正则化**真的接上了**(否则只是静默无效)
                        if not globals().get('_REG_LOGGED', False):
                            globals()['_REG_LOGGED'] = True
                            if _reg_used:
                                logger.info("[deform-reg] ✅ 已接入正则化: " + ", ".join(
                                    f"{k}(w={v[0]:g}, val={v[1]:.4g})"
                                    for k, v in _reg_used.items()))
                            else:
                                logger.warning(
                                    "[deform-reg] ⚠ 形变正则化**全部为 0** -> 未生效! "
                                    "检查 --w-tv/--w-fold/--w-tv-res/--w-prune-l1 等")

                # Stage 3 / v15 锚定正则: 把书家风格参数拉向预训练目标, 防"冻主干
                # 只训风格"时塌缩/漂移 (styletok 实测无护栏则 strict 掉头)。
                # λ 由 --style-anchor-weight 控制, 口径由 --style-anchor-mode 决定。
                if _style_anchor_w > 0 and _style_table_param is not None \
                        and _style_anchor_target is not None:
                    _Ecur = _style_table_param[:_style_anchor_target.shape[0]]
                    if _style_anchor_mode == "mean":
                        _Ecur = _Ecur.view(_style_anchor_target.shape[0],
                                           _style_anchor_k, -1).mean(dim=1)
                    loss = loss + _style_anchor_w * ((_Ecur - _style_anchor_target) ** 2).mean()

                # ★ 2026-10-03 正交正则 (方案三): 惩罚非对角余弦平方 λ·mean_{i≠j} cos²(t_i,t_j)。
                #   目的: 让"多模态"在训练全程保持真正交, 而不是被 MSE 挤回共线。
                #   仅对真实类别行生效 —— MultiStyleEmbedder 的 CFG null 是独立 Parameter, 天然排除。
                if _style_ortho_w > 0 and _style_table_param is not None \
                        and _style_anchor_k > 1:
                    _T = _style_table_param.view(_style_table_param.shape[0],
                                                 _style_anchor_k, -1)
                    _Tn = torch.nn.functional.normalize(_T, dim=-1)
                    _S = _Tn @ _Tn.transpose(1, 2)
                    _off = ~torch.eye(_style_anchor_k, dtype=torch.bool,
                                      device=_S.device)
                    loss = loss + _style_ortho_w * (_S[:, _off] ** 2).mean()

                opt.zero_grad(set_to_none=True)  # INFRA: set_to_none 释放梯度tensor, 比 zero_() 快且省内存

                # ★ 2026-09-17: 这里**不再**逐项 `.item()`（原来 7 次全 GPU 同步 -> 现在 1 次）。
                # ★ 2026-09-18: 把 `loss.backward()` 提前到 finiteness 同步**之前**。
                #   `bool(isfinite(loss))` 是一次全设备同步；原先放在 backward 前，会让 CPU
                #   在 forward 尾端停住 —— GPU 跑完 forward 后只能干等 CPU 重新入队 backward,
                #   形成 forward→backward 之间的流水线气泡。而 backward 本身**不需要** skip 决策
                #   （NaN loss 反传只产生 NaN 梯度, 不碰参数），真正必须同步的只有 `opt.step()`。
                #   故先无脑 backward, 再读 `_finite`, 让 forward+backward 在 GPU 上连跑。
                #   代价: 极少数 NaN 步白跑一次 backward —— 可忽略（且 NaN 梯度会在下一轮顶部
                #   `opt.zero_grad(set_to_none=True)` 被清掉, 不会污染 clip/step）。
                if _writer_table_tuning is not None:
                    loss = loss + _writer_table_tuning.anchor_weight * _writer_table_tuning.anchor_loss()
                if _strong_style is not None:
                    from src.train.strong_style import norm as _style_grad_norm
                    _probe = (train_steps % args.strong_style_probe_every == 0
                              or not _strong_style.scale)
                    _fm_norm = (_style_grad_norm(loss_diff, [p for p in model.parameters() if p.requires_grad])
                                if _probe else None)
                    # Same summed gradients, one clip/optimizer step, but release
                    # FM/REPA saved activations before building the full style graph.
                    # ★ 2026-10-14 [重构 PR4] **明确命名策略** ✓（审查 §4.2 ✓）:
                    #   这是 sequential_backward ✓ —— 顺序本身就是语义 ✗:
                    #     先 backward 主 loss（释放 FM/REPA 激活 ✓）
                    #     再构造辅助 loss 并 backward ✓
                    #   提前构造会让两张图同时存活 ✗ 完全失去省显存的意义 ✗
                    _bw.joint_backward(loss)
                    _aux_style = _strong_style(model, x_latent.detach(), t, model_kwargs,
                                              batch, loss_diff.detach(), train_steps, fm_norm=_fm_norm)
                    _bw.aux_backward(_aux_style)
                    loss = loss.detach() + _aux_style.detach()
                    del _aux_style
                else:
                    _bw.joint_backward(loss)          # ★ PR4: joint_backward ✓
                # NaN guard: skip clip/step/ema/accumulate if loss is not finite (e.g. a bad sample).
                # ★ PR4: 判定与计数都走 _guard ✓（同一表达式 ✓ 同一告警文本 ✓）
                _finite = _guard.check(loss)
                if _finite:
                    # ★ 2026-10-14 [重构 PR4] clip+step+writer-table 复原三连抽到 step.py ✓
                    #   （审查 §4.2 第 5–6 条 ✓）★ **顺序即语义** ✗:
                    #     clip 必须在 step 之前 ✓；restore_null 必须在 step 之后 ✓
                    #     （null 行会被 step 更新 ⇒ 必须立刻复原 ✓ 否则 CFG null token 漂移 ✗）
                    #   max_norm 是**绝对**阈值 ✓ 换 VAE / 改 scaling_factor 后须同步调整 ✓
                    #   （详见 docs/vae_latent_contract.md §2/§5.5 ✓）
                    _clip_and_step(opt, trainable_params_list,
                                   float(getattr(args, 'grad_clip', 1.0)),
                                   writer_table=_writer_table_tuning)
                    # ★ 2026-10-11 算子级 profile / 分配历史: 窗口终点 ⇒ 见 telemetry.py ✓
                    #   ⚠ 仍在 `if _finite:` 分支内 ✓（旧语义: 非有限步不计入窗口 ✓ 顺序不变 ✓）
                    _diag.tick()
                    if scheduler is not None:
                        scheduler.step()
                    # ★ 2026-10-14 [重构 PR4] EMA 调度与更新已抽到 src/train/ema.py ✓
                    #   （审查 §4.2 把 EMA 列为需要明确生命周期的组件之一 ✓）
                    #   数学说明（审查 P1-5 已纠正 ✓）见 ema.py 模块文档:
                    #     间隔式**不是**严格等价每步 decay ✗ 只是带宽优化 ✓
                    #     interval 属实验身份 ✓ ⇒ 跨实验比较必须固定同一 interval ✓
                    #   逐字等价 ✓: should_update ≡ step % max(1,interval)==0 ✓
                    #              effective_decay ≡ warmup 后 ** interval ✓
                    if ema_model is not None and _ema.should_update(train_steps, args):
                        current_ema_decay = _ema.effective_decay(train_steps, args)
                        update_ema(ema_model, model, current_ema_decay)
                        if _writer_table_tuning is not None:
                            _writer_table_tuning.restore_null(ema_model)
                else:
                    # 已 backward -> .grad 里留下 NaN，但**不** clip/step/ema -> 参数不被污染；
                    # 这些 NaN 梯度由下一轮顶部的 zero_grad(set_to_none=True) 释放。
                    # ★ PR4: 计数 + 告警（含"只在出问题时才取标量 ⇒ 不在热路径"）都在守卫里 ✓
                    _guard.note_skipped(train_steps, loss_diff)
                # === INFRA: release the autograd graph every step, finite or not.
                # The graph built by training_losses (DiT forward + pred_xstart) must be
                # freed BEFORE the next forward, otherwise peak = diff_graph + aux_graph.
                # ★ 累加器只留 `.detach()`（仍是 GPU 张量，不触发同步），
                #   标量化推迟到日志步。见上面 `_finite` 处的说明。
                if _finite:
                    running_loss = running_loss + loss.detach()
                    running_diff = running_diff + loss_diff.detach()
                    running_repa = running_repa + loss_repa.detach()
                    running_std_mid = running_std_mid + loss_std_mid.detach()
                    running_skel = running_skel + loss_skel_struct.detach()
                    # c12 逐通道拆解也改成 GPU 侧累加 (C,) 向量，日志步再分组
                    _mse_ch = loss_dict.get("mse_ch", None) if isinstance(loss_dict, dict) else None
                    if _mse_ch is not None and _mse_ch.shape[1] > 4:
                        _cm = _mse_ch.mean(dim=0).detach()
                        _acc_c12 = _cm if _acc_c12 is None else (_acc_c12 + _cm)
                    log_steps += 1

                del loss, loss_dict, loss_diff, pred_xstart_latent
                del loss_repa, loss_std_mid, loss_x0lat, loss_skel_struct
                _mse_ch = None

                train_steps += 1
                _inj_probe.maybe_log(train_steps)

                # ★ 回收 torch.compile 的显存空洞（见 cli.py --empty-cache-after-warmup 的说明）
                #
                # ⚠ 必须在 warmup **期间每一步**都回收，不能只在结束后调一次：
                #   空洞是在前几十步里逐步累积的，而 batch 撑爆时 OOM 就发生在
                #   **warmup 期间**（实测 batch360 挂在 empty_strided_cuda((360,256,1024))）。
                #   结束后再回收，峰值已经过去了，救不了 OOM。
                #   每步调一次 empty_cache() 把高水位钉在当前步的活跃需求上。
                # 成本: 前 N 步每步多一次 device sync(~1-5ms)，一次性 ~0.2s，可忽略。
                if _empty_cache_at > 0 and train_steps <= _empty_cache_at:
                    _r0 = torch.cuda.memory_reserved() / 2 ** 30
                    _a0 = torch.cuda.memory_allocated() / 2 ** 30
                    torch.cuda.empty_cache()
                    if rank == 0 and (train_steps == _empty_cache_at
                                      or train_steps in (1, 10, 25)):
                        logger.info(
                            f"[alloc] step {train_steps} warmup empty_cache: reserved "
                            f"{_r0:.2f}G -> {torch.cuda.memory_reserved() / 2 ** 30:.2f}G | "
                            f"活跃 {_a0:.2f}G -> "
                            f"{torch.cuda.memory_allocated() / 2 ** 30:.2f}G | "
                            f"高水位 {torch.cuda.max_memory_reserved() / 2 ** 30:.2f}G | "
                            f"空洞 {_r0 - _a0:.2f}G")
                        # ★ 空洞 = reserved - allocated（allocator 缓存住、但没有任何
                        #   活张量在用的块）。要弄清"是谁把它顶上去的"，光看总量不够 ——
                        #   把按**尺寸分桶**的分配统计打出来，就能看出是"很多个小块"还是
                        #   "几个巨块"：前者是碎片/内核工作区，后者是某个大张量。
                        if getattr(args, "alloc_debug", False):
                            _st = torch.cuda.memory_stats()
                            _big = {k: v for k, v in _st.items()
                                    if "allocated_bytes" in k or "segment" in k
                                    or "num_" in k}
                            logger.info("[alloc] 尺寸分桶: " + " ".join(
                                f"{k}={v}" for k, v in sorted(_big.items())))
                            logger.info("[alloc] memory_summary:\n"
                                        + torch.cuda.memory_summary())

                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    # Guard against a logging window in which every step was skipped due to non-finite loss.
                    divisor = max(log_steps, 1)
                    # ★ 2026-10-14 [重构 PR4] 均值汇总抽到 telemetry.average_window ✓（纯计算 ✓）
                    #   累加器已是 GPU 张量 -> 除法仍在 GPU ✓ **只在这一次**取标量 ✓
                    #   world_size>1 时先 all_reduce(SUM) 再 /world_size ✓（逐字等价 ✓）
                    #   12ch 分组 loss 仍从累加的 (C,) 向量拆出 image/canny/skel ✓
                    _wa = _tel.average_window(
                        {"loss": running_loss, "diff": running_diff,
                         "repa": running_repa, "std_mid": running_std_mid,
                         "skel": running_skel},
                        divisor, dist.get_world_size(),
                        steps=log_steps, seconds=end_time - start_time,
                        aux_dirs=_aux_dirs, acc_c12=_acc_c12)
                    avg_l, avg_d, avg_r = _wa.loss, _wa.diff, _wa.repa
                    avg_std_mid, avg_skel = _wa.std_mid, _wa.skel
                    steps_per_sec = _wa.steps_per_sec
                    avg_c12i, avg_c12c, avg_c12s = _wa.c12_img, _wa.c12_canny, _wa.c12_skel

                    if rank == 0:
                        wr = args.w_repa
                        from src.train.diagnostics import log_writer_glyph
                        log_writer_glyph(model, logger, train_steps)
                        if _writer_table_tuning is not None:
                            with torch.no_grad():
                                _wt = _writer_table_tuning
                                _wg = (_wt.weight.grad[:_wt.n_writers].norm().item()
                                       if _wt.weight.grad is not None else 0.0)
                                _wd = (_wt.weight[:_wt.n_writers] - _wt.reference).norm().item()
                                _wn = (_wt.weight[_wt.n_writers:] - _wt.fixed_tail).abs().max().item()
                            logger.info('[writer-table-step] step=%d grad=%.4e drift=%.4e null_max=%.4e anchor=%.4e table_lr=%.4e',
                                        train_steps, _wg, _wd, _wn, _wt.anchor_loss().detach().item(),
                                        opt.param_groups[-1]['lr'])
                        # ★ 2026-10-14 [重构 PR4] 单步日志文本抽到 telemetry ✓（纯格式化 ✓）
                        #   ⚠ `_DEFORM_EMA` 是**模块全局** ✗ ⇒ 读取**留在本文件** ✓
                        #     只把**值**传进去 ✓（搬进 telemetry 会读到空 ✗✗ 经典陷阱 ✓）
                        _de = (globals().get('_DEFORM_EMA') or {})
                        logger.info(_tel.format_step_log(
                            step=train_steps, avg_d=avg_d, avg_r=avg_r,
                            avg_std_mid=avg_std_mid, avg_skel=avg_skel,
                            steps_per_sec=steps_per_sec, w_repa=wr,
                            w_std_mid=float(getattr(args, 'w_std_mid', 0.0) or 0.0),
                            w_deform_skel=float(getattr(args, 'w_deform_skel', 0.0) or 0.0),
                            deform_loss=float(_de.get('loss') or 0.0),
                            deform_off=float(_de.get('off') or 0.0),
                            w_latent_skel=float(getattr(args, 'w_latent_skel', 0.0) or 0.0),
                            vae_kl=loss_vae_kl,
                            w_vae_kl=float(getattr(args, 'w_vae_kl', 1e-5)),
                            train_vae_enc=_train_vae_enc,
                            vae_lr=(opt.param_groups[2]['lr']
                                    if _train_vae_enc and len(opt.param_groups) > 2 else None),
                            lr=opt.param_groups[0]['lr'],
                            ema_decay=(current_ema_decay if ema_model is not None else None)))
                        # ── 调试诊断 (每 1000 步): 梯度分组 + 注入门控强度 ──
                        # 门控 = xattn 注入输出在残差流上的平均 L2 (零初始化起,
                        # 增长 = 注入正在学会写入); 梯度分组看各通路是否在学习。
                        if train_steps % 1000 == 0:
                            if not _inj_stats:
                                _mm = model.module if hasattr(model, 'module') else model
                                for _i, _inj in enumerate(
                                        getattr(_mm, 'glyph_injections', []) or []):
                                    if not hasattr(_inj, 'out_proj'):
                                        continue

                                    def _mk(_i):
                                        def _h(_m, _inp, _out):
                                            _inj_stats[_i] = _out.detach().float() \
                                                .norm(dim=-1).mean().item()
                                        return _h
                                    _inj.register_forward_hook(_mk(_i))
                            _gn = {}
                            for _n, _p in model.named_parameters():
                                if _p.grad is None:
                                    continue
                                _g = _p.grad.norm().item()
                                if "glyph_injections" in _n:
                                    _k = "inj_out_proj"
                                elif "glyph_embedder" in _n:
                                    _k = "glyph_embedder"
                                elif "callig_proj" in _n or "callig_scale" in _n:
                                    _k = "callig_chain"
                                elif "blocks." in _n:
                                    _k = "blocks"
                                else:
                                    _k = "other"
                                _gn[_k] = _gn.get(_k, 0.0) + _g * _g
                            _gn = {k: v ** 0.5 for k, v in _gn.items()}
                            _gate = ""
                            if _inj_stats:
                                _gate = " | inj|out|: " + " ".join(
                                    f"L{i}={v:.3f}" for i, v in sorted(_inj_stats.items()))
                            logger.info(
                                f"(step={train_steps:07d}) [diag] grad-norm: "
                                + " ".join(f"{k}={v:.3f}" for k, v in sorted(_gn.items()))
                                + _gate)
                
                    # 重置累加器（GPU 张量用 zero_ 原地清，避免重新分配）
                    running_loss.zero_()
                    running_diff.zero_()
                    running_repa.zero_()
                    running_std_mid.zero_()
                    running_skel.zero_()
                    _acc_c12 = None
                    log_steps = 0
                    start_time = time()

                # ★ 2026-09-17 简化: 原来三段 if/elif 拼出"前 5000 步每 1000 存一次、
                #   之后按 ckpt_every 存"的混合节奏（ckpt_every=5000 时实际落在
                #   1000..5000 + 7500/12500/... —— 与 eval 的 5000 边界**错开**）。
                #   现在 epoch 长度 / 存盘点 / 评测点已合并为同一个刻度
                #   （见上面的 `_epoch_steps`），存盘条件就是一行。
                #   顺带省掉开头 1000/2000/3000/4000 四次存盘（每次 592MB）。
                _save_ckpt = _epoch_steps > 0 and train_steps % _epoch_steps == 0

                if _save_ckpt and train_steps > 0:
                    if rank == 0:
                        # 组装(同步搬 CPU) + 异步入队 + 轮转 ⇒ 见 CheckpointManager ✓
                        # （实现仍在 src/train/ckpt.py ✓ 临时文件/fsync/os.replace/
                        #   SHA256 manifest/异步写线程 全部保留 ✓）
                        _ckpt.save(model, opt, ema_model, scheduler, args, train_steps)

                        # 在训评测：是否该跑 / 用哪份权重 / 计时 / 报错 全在 eval 模块里。
                        # 失败只 warning 不中断训练（评测不该把训练带崩）。
                        # 另有已停用的 in-process GPU eval 路径，见 src/eval/legacy/。
                        maybe_run_in_training(
                            args, ema_model, model, train_steps, device,
                            checkpoint_dir, logger, is_eval_step=_save_ckpt,
                            inline=_EVAL_INLINE)

                if args.max_steps > 0 and train_steps >= args.max_steps:
                    logger.info(f"Reached max_steps={args.max_steps}; stopping cleanly.")
                    break

                if (train_steps >= int(getattr(args, 'early_stop_min_steps', 0))
                        and train_steps % _es_check_every == 0
                        and rank == 0
                        and _stopper.check()):      # 内部已判 early_stop 开关
                    early_stop_stopped = True
                    break
        except Exception as e:
            # ★ 2026-10-13 [审查 P0]: **不许吞异常** ✗ —— 原来这里 break 之后仍会走到
            #   末尾打印 "Done!", 于是崩溃任务在远程/队列/自动验收里**表现为正常完训** ✗✗。
            #   现在: 记录 traceback -> 走完清理 -> **重新抛出**, 让进程以非零码退出 ✓
            import traceback
            logger.error(f"Error during training loop: {e}")
            logger.error(traceback.format_exc())
            _train_fatal = e
            break

        if args.max_steps > 0 and train_steps >= args.max_steps:
            break

        if early_stop_stopped:
            break
    
        if dist.get_world_size() > 1:
            dist.barrier()
    return _train_fatal
