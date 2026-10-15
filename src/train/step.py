# -*- coding: utf-8 -*-
"""step.py — **单次更新的组件**（2026-10-14 [重构 PR4] ✓ 第一块）

审查 §4.2 原文（单步训练应独立管理）:
  "1. H2D 数据移动；2. Flow Matching 的 forward；3. REPA / Critic / 几何损失；
   4. backward；**5. finite 梯度检查和 clipping**；6. optimizer / scheduler / EMA 更新。" ✓

本文件先落地第 5 条里**最自包含**的一件 ✓: **非有限 loss 的跳过策略** ✓
（`clip` 本身仍由调用方执行 ✓ —— 因为它需要 `trainable_params_list` 与 `args.grad_clip` ✓，
 属"训练循环的上下文" ✓ 本模块不接管 ✗）

★ 为什么先抽它 ✗: 它是循环体里少见的**纯状态机** ✓
  · 输入: 本步的 loss ✓ 步数 ✓
  · 状态: 累计跳过步数 ✓
  · 输出: 本步**是否允许** clip/step/ema ✓
  · 副作用: 只在**出问题时**打一条 warning ✓
  ⇒ 边界清晰 ✓ 可单测 ✓ 且搬迁后**行为逐字不变** ✓

★ 语义（原文逐字保留 ✓）:
  "NaN guard: skip clip/step/ema/accumulate if loss is not finite (e.g. a bad sample)." ✓
  "已 backward -> `.grad` 里留下 NaN，但**不** clip/step/ema -> 参数不被污染；
   这些 NaN 梯度由下一轮顶部的 `zero_grad(set_to_none=True)` 释放。" ✓
  ⚠ 注意: backward **已经发生** ✓ 本模块只决定"**要不要 step**" ✓
    —— 它**不**撤销 backward ✗ 也**不**清梯度 ✗（那是下一轮 zero_grad 的事 ✓）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["NonFiniteGuard", "StepInputs", "resolve_step_inputs",
           "clip_and_step"]


class NonFiniteGuard:
    """**非有限 loss 的跳过策略** ✓（审查 §4.2 第 5 条 ✓）。

    用法（与旧的两处内联**逐字等价** ✓）::

        guard = NonFiniteGuard(logger, rank)
        ...
        if guard.check(loss):
            clip_grad_norm_(...); opt.step(); ...          # 正常路径 ✓
        else:
            guard.note_skipped(train_steps, loss_diff)     # 计数 + 告警 ✓
    """

    def __init__(self, logger=None, rank: int = 0):
        self.logger = logger
        self.rank = int(rank)
        self.nan_steps = 0

    def check(self, loss) -> bool:
        """本步 loss 是否**有限** ✓ —— 逐字等价于旧的 `bool(torch.isfinite(loss))` ✓。

        ⚠ 这一步**会**同步（分支依赖它的值 ✓ 无法避免 ✓）；
          旧实现也是同一表达式 ✓ 行为不变 ✓。
        """
        return bool(torch.isfinite(loss))

    def note_skipped(self, step: int, loss_diff=None) -> None:
        """记一次跳过 ✓ + **仅 rank0** 打一条 warning ✓（文本与旧实现逐字一致 ✓）。

        ⚠ `float(loss_diff)` 只在**出问题时**才求值 ✓ ⇒ **不在热路径** ✓
          （旧注释原话: "这里必须取标量（要打出来），但只在**出问题时**才发生 -> 不在热路径" ✓）
        """
        self.nan_steps += 1
        if self.rank != 0 or self.logger is None:
            return
        _diff_txt = ""
        if loss_diff is not None:
            _diff_txt = f"(diff={float(loss_diff):.4f})"
        self.logger.warning(
            f"Step {step} skipped: non-finite loss "
            f"{_diff_txt}. Accumulated skips: {self.nan_steps}"
        )

    def stats(self) -> dict:
        """状态摘要 ✓（供收尾日志/manifest ✓）。"""
        return {"nan_steps": int(self.nan_steps)}

    def describe(self) -> str:
        return f"non_finite_skips={self.nan_steps}"

# ══════════════════ 单步输入: batch → (x_latent, x, loss_vae_kl) ══════════════════
@dataclass
class StepInputs:
    """单步训练所需的输入张量 ✓（纯数据 ✓）。"""
    x_latent: torch.Tensor
    x: Optional[torch.Tensor]
    loss_vae_kl: Optional[torch.Tensor]


def resolve_step_inputs(batch, vae, device, *, train_vae_enc: bool,
                        vae_sf: float, vae_shift: float) -> StepInputs:
    """把 batch 解析成扩散目标 `x_latent`（审查 §4.2 第 1 条: H2D 数据移动 ✓）。

    **逐字搬移**自 `train.py` 的三分支 ✓（连上下文管理器一起 ✓）:
      ① `train_vae_enc`  —— 实时读图 + VAE `rsample()` ✓
           ⚠ **必须** `rsample()` ✗: 只有它能让 DiT 梯度穿过高斯采样流回 VAE Encoder ✓
           ⚠ **不套** `no_grad` ✓（要训 VAE ✓）；latent 合同: `(z_raw - SHIFT) * SF` ✓（合同 I2 ✓）
      ② `latent` 缓存    —— 预编码 latent ✓（已按 SF 缩放 ✓）+ aux 通道拼接 ✓
      ③ 其它             —— 实时读图 + VAE `sample()` ✓（`no_grad` + fp32 autocast ✓
                              VAE 对低精度敏感 ✓）
    ★ 三条分支的**梯度语义不同** ✗ ⇒ 本函数刻意**不**统一它们 ✓
    """
    if train_vae_enc and "image" in batch:
        # ⭐️ [REPA-E] 必须实时读取真实图像，使用 rsample() 打通反向传播梯度管道
        x = batch["image"].to(device, non_blocking=True)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0).mul_(2.0).sub_(1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            posterior = vae.encode(x).latent_dist
            # ⚠ 必须使用 rsample() 重参数化采样: z = mu + sigma * eps
            # 只有 rsample 才能让 DiT 的梯度穿过高斯采样，流回 VAE Encoder 的权重！
            # ★ 2026-10-10: 必须与 **shards 的存盘约定**一致 ——
            #   shards = (z_raw - SHIFT) * SF（合同 I2）; 这里原来只 mul_(sf) 漏了 shift,
            #   与预编码路径不一致(2016-10-10 发现)。加 _vae_shift 保持两条路同约定。
            x_latent = (posterior.rsample() - vae_shift).mul_(vae_sf).float()
            # 流形的物理护栏: 计算隐空间高斯 KL 散度
            loss_vae_kl = -0.5 * torch.sum(
                1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp(),
                dim=[1, 2, 3]
            ).mean()
        return StepInputs(x_latent=x_latent, x=x, loss_vae_kl=loss_vae_kl)
    if "latent" in batch:
        # Latent-cached training: latent pre-encoded (scaled by vae_scaling_factor).
        x_latent = batch["latent"].to(device, non_blocking=True)
        # moyi 式辅助目标通道: aux latents (skel/canny) 与图像 latent 拼成扩散目标
        _aux = batch.get("aux_latents", None)
        if _aux is not None and _aux.numel() > 0:
            x_latent = torch.cat(
                [x_latent, _aux.to(device, non_blocking=True).float()], dim=1)
        x = batch.get("image", None)
        x = x.to(device, non_blocking=True) if x is not None else None
        return StepInputs(x_latent=x_latent, x=x, loss_vae_kl=None)
    x = batch["image"].to(device, non_blocking=True)
    # dataset 现在给的是 uint8（省 H2D 流量），VAE 需要 [-1,1] float
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0).mul_(2.0).sub_(1.0)
    # VAE encode stays in fp32 for numerical stability (VAE is sensitive to low precision).
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
        x_latent = (vae.encode(x).latent_dist.sample() - vae_shift).mul_(vae_sf)
        x_latent = x_latent.float()
    return StepInputs(x_latent=x_latent, x=x, loss_vae_kl=None)

def clip_and_step(opt, params, max_norm: float, writer_table=None) -> None:
    """成功反传后的**三连** ✓（审查 §4.2 第 5–6 条: finite 检查 / clip / 更新 ✓）。

    ★ **顺序即语义** ✗:
      ① `clip_grad_norm_` 必须在 `opt.step()` **之前** ✓（否则等于没裁 ✓）
      ② `writer_table.restore_null()` 必须在 `opt.step()` **之后** ✓
         —— writer table 的 **null 行**会被 step 更新 ✓ ⇒ 必须立刻复原 ✓
            否则 CFG 的 null token 会跟着漂移 ✗（那是**语义**错误 ✗ 不只是数值问题 ✓）

    ⚠ `max_norm` 是**绝对**阈值 ✓（原注释）:
        "梯度幅度随数据端 latent 尺度**线性缩放** ⇒ 换 VAE / 改 scaling_factor
         后必须同步调整, 否则隐性裁剪。默认 1.0 保持历史行为。"
        （详见 docs/vae_latent_contract.md §2/§5.5 ✓）

    ⚠ 本函数**只**做这三件事 ✓ —— 不含 `zero_grad` ✗ 不含 scheduler/EMA ✗
      （那些的时机与条件由调用方掌握 ✓ 审查原则 B ✓）
    """
    torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_norm))
    opt.step()
    if writer_table is not None:
        writer_table.restore_null()
