# -*- coding: utf-8 -*-
"""sampler.py — **唯一的采样实现**（2026-10-13 [审查 §4.1] 从 src/eval/inference.py 迁出）

审查迁移表: "inference.py:38-313 -> sampler.py：**保留唯一 Heun/SDE/CFG 实现**" ✓

本模块只负责"条件 + 噪声 -> latent"这一步:
  · build_diffusion              : 构造 diffusion/flow 对象（ODE/DDIM 步数 ✓）
  · model_uses_std_soft          : 该模型是否走 std-soft 空间条件 ✓
  · validate_std_soft_condition  : std-soft 必须收到非空软掩码（否则**报错**, 不静默兜底 ✗）
  · sample_latents               : 主采样（CFG 由**模型层** forward_with_cfg 处理 ✓）
  · sample_latents_self_cond     : 两遍自条件采样 ✓

⚠ 迁移策略（审查"第一阶段只移动逻辑" ✓）: 这些名字仍由 src/eval/inference.py
  **原样再导出**（同一函数对象 ✓）⇒ 既有调用点无需改动 ✓✓。

刻意**不**依赖 inference（否则成环 ✗）: 只依赖 src.eval.decoder + src.loss ✓
"""
from __future__ import annotations

import torch
import torch as th

from src.loss import create_diffusion_or_flow
from src.eval.decoder import match_model_channels
from src.model.conditioning import build_batch, kwargs_for_call

__all__ = ["build_diffusion", "model_uses_std_soft", "validate_std_soft_condition",
           "model_kwargs_for_slice",
           "sample_latents", "sample_latents_self_cond"]


# ── 扩散对象 ─────────────────────────────────────────────────────────────────
def build_diffusion(steps, diffusion_type="ddpm", flow_kwargs=None):
    """Build the diffusion sampler for eval. steps: int (ODE/DDIM step count).

    ``flow_kwargs`` (dict) 透传给 FlowMatching —— 例如
    ``{"t_sampler": "logit_normal", "sampler": "heun", "shift": 1.0}``。
    推理侧必须与训练侧使用同一份 flow 配置，否则会静默地用另一套
    时间步分布 / 求解器去评估模型。
    """
    return create_diffusion_or_flow(str(steps), diffusion_type=diffusion_type,
                                    **(flow_kwargs or {}))


# ── 采样 (GPU, bf16) → latents on CPU ───────────────────────────────────────
def model_uses_std_soft(model):
    """Recognize plain, compiled, DDP and ControlNet-wrapped std-soft models."""
    pending, seen = [model], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if bool(getattr(current, "std_soft_cond", False)):
            return True
        pending.extend(getattr(current, key, None)
                       for key in ("main", "module", "_orig_mod"))
    return False


def validate_std_soft_condition(g, n):
    """Fail closed: missing/blank std-soft must never produce valid-looking scores."""
    if not isinstance(g, torch.Tensor) or tuple(g.shape) != (n, 1, 64, 64):
        shape = None if g is None else getattr(g, "shape", type(g).__name__)
        raise ValueError(f"std-soft evaluation requires nonempty g [N,1,64,64]; got {shape}")
    if n <= 0 or not g.is_floating_point() or not torch.isfinite(g).all():
        raise ValueError("std-soft condition must be finite floating-point masks")
    if (g < 0).any() or (g > 1).any():
        raise ValueError("std-soft condition must use ink=1/background=0 in [0,1]")
    if (g.flatten(1).sum(1) <= 0).any():
        raise ValueError("std-soft condition contains blank glyphs")


@torch.no_grad()

def model_kwargs_for_slice(model, *, yc, yh, hier_slice=None, skel_slice=None,
                           device="cpu") -> dict:
    """装配一次 `ddim_sample_loop` 的 `model_kwargs` ✓（2026-10-14 [重构 PR5] ✓）

    ★ 逐字搬移自 `sample_latents` 的内联逻辑 ✓（审查 §4.1: 收进 `ConditionBatch` ✓）
    抽成函数只为**可测** ✓ —— 逻辑与顺序一字未改 ✓。

    ★★ 关键契约（**已实测** ✓ 见 tests/test_refactor_migration_rule.py ✓）:
      必须用 `kwargs_for_call` ✗ —— **不能**用 `batch_to_kwargs(for_forward=True)` ✓
        后者**总是**保留 `g`（契约: `g=None` = "**条件被丢掉**" ✓）
        ⇒ 会把"没传 g"变成"明确传 g=None" ✗ ⇒ **语义改变** ✗
      本函数用 `kwargs_for_call` ✓ ⇒ 与旧的"**存在才加键**"逐字等价 ✓

    ★ `cond` 键（ControlNet 包装 ✓）**不在** `ConditionBatch` 里 ✗
      ⇒ 按契约**留在调用方**（本函数内）✓ 不要硬塞进兼容层 ✗

    参数:
      hier_slice : `(hraw, hpair, hscript)` 的**已切片**序列 ✓ 或 None ✓
      skel_slice : 字形/骨架张量 ✓ 或 None ✓（None ⇒ **根本不传**该键 ✓）
    """
    _hraw = _hpair = _hscript = None
    if hier_slice is not None:
        _hraw, _hpair, _hscript = hier_slice
    # ── S2 三层语义分解的三个 id（None = 模型走旧路径，零副作用）──────────
    _batch = build_batch(
        y_callig=yc, y_char=yh,
        y_callig_raw=(torch.tensor(_hraw, device=device, dtype=torch.long)
                      if _hraw is not None else None),
        y_pair=(torch.tensor(_hpair, device=device, dtype=torch.long)
                if _hpair is not None else None),
        y_script=(torch.tensor(_hscript, device=device, dtype=torch.long)
                  if _hscript is not None else None))
    mk = kwargs_for_call(_batch)
    if skel_slice is not None:
        # g 条件模型 (train.py use_glyph_cond/skel_as_glyph_cond/std_soft_cond 预训练) 走 'g' 键;
        # ControlNetDiT 包装走 'cond' 键。
        # Recognize wrappers and the accepted keyword; flags alone are not
        # sufficient for every historical checkpoint/model wrapper.
        import inspect as _inspect
        _fn = getattr(model, "forward_with_cfg", None) or model
        try:
            _params = set(_inspect.signature(_fn).parameters)
        except Exception:                                              # noqa: BLE001
            _params = set()
        _use_g = (getattr(model, "use_glyph_cond", False)
                  or getattr(getattr(model, "main", None), "use_glyph_cond", False)
                  or getattr(model, "std_soft_cond", False)
                  or getattr(getattr(model, "main", None), "std_soft_cond", False)
                  or model_uses_std_soft(model)
                  or ("g" in _params and "cond" not in _params))
        if _use_g:
            mk["g"] = skel_slice.to(device)
        else:
            # ⚠ `cond` 是 ControlNet 包装的键 ✗ 不在 ConditionBatch 里 ✓
            #   ⇒ 按契约留在调用方 ✓（不要硬塞进兼容层 ✗）
            mk["cond"] = skel_slice.to(device)
    return mk

def sample_latents(model, diffusion, noise, conds, cfg_scale, batch, device,
                   skel=None, seed=0, hier_conds=None,extra_conditions=None):
    """Run diffusion sampling (DDIM for ddpm / Euler for flow) → CPU latents.

    model       : callable(x, t, y_callig, y_char, cfg_scale=..., cond=...) —
                  i.e. ``model.forward_with_cfg`` (CFG handled at the model level).
    noise       : (N, C, H, W) fixed noise (CPU).
    conds       : list of (callig_id, glyph_id) tuples, length N.
    cfg_scale   : classifier-free guidance scale (0 = no CFG).
    skel        : optional structural condition (ControlNet path):
                  (N,4,32,32) VAE latent (新) 或 (N,1,256,256) PNG (旧).
    Returns (N, C, H, W) float32 latents on CPU.
    """
    n = noise.shape[0]
    if model_uses_std_soft(model):
        # v83-A remote eval used an old cache returning None while training had g.
        validate_std_soft_condition(skel, n)
    noise = match_model_channels(noise, model)   # aux 目标通道: 噪声扩到 in_channels
    lc, ls = noise.shape[1], noise.shape[2]
    all_latents = torch.zeros(n, lc, ls, ls, dtype=torch.float32)
    torch.manual_seed(seed)  # deterministic per call; noise itself is fixed anyway
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    # CFG 在模型层处理 (forward_with_cfg)。不能把 cfg_scale 塞进 model_kwargs:
    # sampler 只做 model(x, t, **kwargs) 转发, plain forward 收到 cfg_scale 会
    # 直接 TypeError (base 通道崩溃), ctrl 通道则静默吞掉 → CFG 从未生效。
    if cfg_scale and cfg_scale > 0:
        # [2026-09-17] 双轴 CFG: 由**模型属性**驱动, 避免改 sample_latents 签名。
        #   cfg_glyph_scale 为 None -> 走经典 2 路 (向后兼容)
        #   非 None -> forward_with_cfg 内部委托给 forward_with_2axis_cfg
        #   ⚠ 仅在训练时 glyph_drop_prob>0 的 ckpt 上才有意义
        #     (g=0 必须是训练见过的条件, 否则内容轴未训练)
        _cg = getattr(model, "cfg_glyph_scale", None)
        _wi = float(getattr(model, "cfg_w_inter", 0.0) or 0.0)

        def model_fn(x, t, **kw):
            extra = {}
            if _cg is not None:
                extra["cfg_glyph"] = float(_cg)
                extra["w_inter"] = _wi
            return model.forward_with_cfg(x, t, cfg_scale=cfg_scale, **extra, **kw)
    else:
        model_fn = model
    # ★★ 2026-10-16 [`infra-1015-measurements` §3 · 用户指示"让评测尽可能快" ✓]:
    #   逐批循环的**三处 launch/同步开销**（纯代码分析定位 ✓ 无需 GPU 实验 ✓）:
    #
    #   ① **每批一次 `torch.cuda.empty_cache()`** ✗✗（原 L201）—— 最恶劣 ✓:
    #      它强制 **设备同步** + 把缓存分配器里的空闲块**全部还给驱动** ✗
    #      ⇒ 下一批的所有分配重新走 `cudaMalloc` ✗（慢一个量级 ✓）
    #      ⇒ 且在**训练内评测**时 ✗ 连**训练侧**的缓存块也被放掉 ✗
    #        ⇒ 评测结束后的第一步训练要重新分配 ✗✗（拖慢训练本身 ✓）
    #      ★ 依据: 训练循环只在 **warmup 第 1 步**做一次 empty_cache ✓
    #        （trainer.py "[alloc] step 1 warmup empty_cache" ✓ —— 正因它代价大 ✓）
    #      ⇒ **彻底删除** ✓（缓存分配器自己会复用空闲块 ✓ 不需要人工干预 ✓）
    #
    #   ② **变长尾批触发 torch.compile 重编译** ✗:
    #      n=187 · batch=8 ⇒ 尾批 shape=(3,) ✗ ≠ 8 ⇒ Dynamo 为每个新 shape 编译一张图 ✗
    #      （且评测 shape 本就 ≠ 训练的 288 ⇒ 首次评测必编译 ✓）
    #      ⇒ **尾批 pad 到固定 batch** ✓（复制最后一样本填充 ✓ 输出丢弃 ✓）
    #        ⇒ 全程只有**一种** shape ✓ 编译一次跨批复用 ✓✓
    #
    #   ③ **逐批 2 次 Python 列表推导建标签张量** ✗（原 L178-179）
    #      ⇒ `conds` 的两列**预建一次** ✓ 之后每批只是切片 ✓
    #
    #   ★★ 结果等价性（必须说清 ✓）:
    #     `noise` 是**固定张量** ✓（cache 预生成 ✓）且 FM 采样**各样本独立** ✓
    #     （DiT 全部用 LayerNorm ✓ 无 BatchNorm ✓ 无跨样本耦合 ✓）
    #     ⇒ 改 batch / pad 只改**切分方式** ✓ **分布不变** ✓
    #     ⚠ 位级可能因 GEMM 归约顺序对不同 M 选不同 kernel 而有极小差 ✗（如实说明 ✓）
    #       ⇒ 用 `in_mem_eval_batch` 调大批后，历史评测数字在**容差内**一致 ✓
    #         （严格逐位对拍请用旧 batch 重跑 ✓ 文档已注明 ✓）
    def _pad_to_batch(t, k):
        """把尾批 pad 到固定 `batch` ✓（复制最后一样本 ✓ 确定性 ✓ 输出会被丢弃 ✓）。"""
        pad = batch - k
        if pad <= 0:
            return t
        return th.cat([t, t[-1:].repeat(pad, *[1] * (t.dim() - 1))], 0)

    # ③ 标签**预建一次** ✓（原每批两次列表推导 ✗）
    _yc_all = torch.tensor([c[0] for c in conds], dtype=torch.long)
    _yh_all = torch.tensor([c[1] for c in conds], dtype=torch.long)
    for i in range(0, n, batch):
        j = min(i + batch, n)
        k = j - i
        z = noise[i:j].to(device)
        if k < batch and dev_type == "cuda":
            # ② 尾批 pad ✓（仅 cuda ✓ CPU 无编译问题 ✓ 不必 pad ✓）
            z = _pad_to_batch(z, k)
        yc = _yc_all[i:j].to(device)
        yh = _yh_all[i:j].to(device)
        # Slice once, pad local tensors, never replace the full condition arrays.
        hier_use = (tuple(th.as_tensor(axis[i:j], dtype=th.long, device=device)
                          for axis in hier_conds) if hier_conds is not None else None)
        if k < batch and dev_type == "cuda":
            yc = _pad_to_batch(yc, k)
            yh = _pad_to_batch(yh, k)
            if hier_use is not None:
                hier_use = tuple(_pad_to_batch(axis, k) for axis in hier_use)
            if skel is not None and th.is_tensor(skel):
                skel_use = _pad_to_batch(skel[i:j], k)
            else:
                skel_use = (skel[i:j] if skel is not None else None)
        else:
            skel_use = (skel[i:j] if skel is not None else None)
        # ★ 2026-10-14 [重构 PR5] 条件装配抽成 model_kwargs_for_slice ✓（审查 §4.1 ✓）
        #   逐字等价 ✓（含"存在才加键"语义 ✓ 与 g/cond 包装器判定 ✓）
        mk = model_kwargs_for_slice(
            model, yc=yc, yh=yh,
            hier_slice=hier_use,
            skel_slice=skel_use,
            device=device)
        if extra_conditions:
            for name,value in extra_conditions.items():
                piece=value[i:j].to(device)
                if k<batch and dev_type=='cuda':piece=_pad_to_batch(piece,k)
                mk[name]=piece
        # bf16 autocast 仅 cuda (CPU 无 AVX512-BF16/AMX 时 bf16 走软件上转反而慢)
        if dev_type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                samples = diffusion.ddim_sample_loop(
                    model_fn, z.shape, z,
                    clip_denoised=False, model_kwargs=mk, device=device)
        else:
            samples = diffusion.ddim_sample_loop(
                model_fn, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device)
        # ①★ `empty_cache()` **已删除** ✓（见函数头说明 ✓）
        all_latents[i:j] = samples[:k].float().cpu()      # ★ pad 部分丢弃 ✓
        del z, samples
    return all_latents


# ── 两遍自条件采样 (Self-Conditioning) ───────────────────────────────────────
# 核心思路: 12ch 联合模型在推理时同时去噪 image/canny/skel 三组通道。
# 其中 skel 通道 (ch 8-11) 的预测实质是「模型认为这个书家会怎么写这个字的骨架」
# —— 即一个 **书家风格化骨架**，比通用印刷标准骨架 g 更接近 GT 骨架。
#
# 将 pass-1 预测的 skel 通道回灌为 pass-2 的条件 g，让模型在第二遍时拿到
# 一个自洽的、带书家个性的骨架条件，而非千篇一律的楷体印刷骨架。
#
# 开销: 约 1.4× (pass-1 用较少 ODE 步) 到 2× (pass-1 同等步数)。
# 不改训练，纯推理侧增益。

@torch.no_grad()
def sample_latents_self_cond(
    model, diffusion, noise, conds, cfg_scale, batch, device,
    skel=None, seed=0,
    hier_conds=None,
    # ── self-cond 参数 ──
    skel_ch_start=8, skel_ch_end=12,
    first_pass_steps=None,
    blend_alpha=0.0,
):
    """两遍自条件采样 (Self-Conditioning) for 12ch joint models.

    Pass 1: 用标准骨架 g 正常采样 → 得到 12ch 预测 (含 skel ch)
    Pass 2: 用 pass-1 预测的 skel 通道替换 g 重新采样 → 更优图像

    Args:
        skel_ch_start, skel_ch_end: 输出中骨架 latent 的通道范围
            (默认 8:12, 即 [img(0-3), canny(4-7), skel(8-11)])。
        first_pass_steps: pass-1 的 ODE 步数 (None = 与 pass-2 相同;
            设较小值如 20 可将总开销从 2× 降至 ~1.4×)。
        blend_alpha: pass-2 条件的混合系数:
            g_pass2 = (1 - α) · predicted_skel + α · g_original
            0.0 = 纯自条件; 0.5 = 各半; 1.0 = 无自条件 (退化为普通采样)。

    Returns:
        (N, C, H, W) float32 CPU latents (与 sample_latents 相同格式)。
    """
    # 非 12ch 模型或无 skel 条件 → 退化为普通采样
    _main = getattr(model, 'main', model)
    model_ch = int(getattr(_main, 'in_channels', 4))
    if model_ch <= 4 or skel is None or blend_alpha >= 1.0:
        return sample_latents(model, diffusion, noise, conds, cfg_scale,
                              batch, device, skel=skel, seed=seed,
                              hier_conds=hier_conds)

    # ── Pass 1: 用标准骨架采样，得到书家风格化骨架预测 ──
    if first_pass_steps is not None and first_pass_steps != diffusion.num_timesteps:
        diff1 = build_diffusion(
            first_pass_steps, diffusion_type='flow',
            flow_kwargs={
                'sampler': getattr(diffusion, 'sampler', 'heun'),
                'heun_batch': getattr(diffusion, 'heun_batch', True),
                'shift': getattr(diffusion, 'shift', 1.0),
            })
    else:
        diff1 = diffusion

    x0_pass1 = sample_latents(model, diff1, noise, conds, cfg_scale,
                              batch, device, skel=skel, seed=seed,
                              hier_conds=hier_conds)

    # 提取预测骨架 (pass-1 的 skel 通道)
    if x0_pass1.shape[1] <= skel_ch_start:
        # 模型输出通道不够 → 无 skel 可提取，退化为普通采样
        return x0_pass1

    predicted_skel = x0_pass1[:, skel_ch_start:skel_ch_end].clone()

    # 混合: 保留部分标准骨架信息 (α > 0 时对预测骨架做保守修正)
    if blend_alpha > 0:
        predicted_skel = (1.0 - blend_alpha) * predicted_skel + blend_alpha * skel

    # ── Pass 2: 用风格化骨架作为条件重新采样 ──
    x0_pass2 = sample_latents(model, diffusion, noise, conds, cfg_scale,
                              batch, device, skel=predicted_skel, seed=seed,
                              hier_conds=hier_conds)
    return x0_pass2
