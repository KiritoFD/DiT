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

import torch

__all__ = ["NonFiniteGuard"]


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
