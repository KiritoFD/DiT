# -*- coding: utf-8 -*-
"""objectives.py — **损失装配（ObjectiveComposer）**（2026-10-14 [重构 PR4] ✓）

审查 §4.2 原文:
  "尤其不要把所有功能都塞进一个通用 `Trainer` 类。我的目标是：`Trainer` 控制循环，
   `TrainStep` 完成一次更新，**`ObjectiveComposer` 组合损失**，`CheckpointManager`
   负责持久化，各自保持小而清晰。" ✓
审查原则 C:
  "实验插件可以增加 Loss 或条件路径，但不能修改核心训练生命周期。后续做
   Writer×Glyph、Critic、REPA 或新空间调制时，**只需注册相应模块和损失**，
   不应该每次在 2,403 行 `main()` 里再插入一个特殊分支。" ✓

本模块只做一件事 ✓: **按注册顺序把各损失项累加** ✓。它
  · **不**调用 optimizer ✗
  · **不**碰 checkpoint ✗
  · **不**决定日志周期 ✗
  · **不**做任何 CUDA→CPU 同步 ✗（`raw` 一律存 **detached tensor** ✓；
    只有显式调用 `as_floats()` 才转数值 ✓ —— 审查 P1-6: "仅在日志周期批量转换" ✓）

★ 为什么强调"**顺序**" ✗:
    浮点加法**不满足结合律** ✗ ⇒ `((a+b)+c) != (a+(b+c))` 在末位上可能不同 ✓
    ⇒ 本模块**按注册顺序**累加 ✓，且调用方必须保持与原代码**相同的注册顺序** ✓
      ⇒ 这样 `total` 才能与原手写链**逐位一致** ✓（见测试 ✓）
    ⇒ **不要**为了让代码好看而重排注册顺序 ✗（那会改变训练轨迹 ✗）
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

__all__ = ["ObjectiveComposer", "compose"]


class ObjectiveComposer:
    """按**注册顺序**累加损失 ✓；如实记录每项 (名字, 权重, raw detached tensor) ✓。

    用法（与手写 `loss = loss + w * term` **逐位等价** ✓）::

        obj = ObjectiveComposer(loss_diff, base_name="fm")
        obj.add("style_rank", loss_style_rank, getattr(args, "w_style_rank", 0.0))
        obj.add("critic_writer", loss_critic_writer, getattr(args, "w_critic_writer", 0.0))
        loss = obj.total

    ⚠ `add` **总是**执行 `total = total + w * loss` ✓ —— **不**因为 `w == 0` 而跳过 ✗。
      这是刻意的 ✓: 与原手写链完全一致 ✓（原链里 `0.0 * x` 也照样加 ✓）
      ⇒ 若 `loss` 是 NaN/Inf ✗，即使权重为 0 也会**照实传播** ✓（不会静默吞掉 ✗）
    """

    def __init__(self, base: torch.Tensor, base_name: str = "fm"):
        self._total = base
        self._names: List[str] = [str(base_name)]
        self._weights: List[float] = [1.0]
        self._raws: List[torch.Tensor] = [base.detach()]

    # ── 累加 ──
    def add(self, name: str, loss: Optional[torch.Tensor], weight: float) -> None:
        """注册并累加一项 ✓（`loss is None` ⇒ 按 0 处理 ✓ 与"项未启用"一致 ✓）。"""
        w = float(weight)
        if loss is None:
            # 与旧代码一致: 未启用时该变量是 `torch.tensor(0.0)` ✓ 权重也常为 0 ✓
            # 这里用 0 张量 + 原权重 ⇒ 数学上等价 ✓（且仍走同一条加法路径 ✓）
            loss = torch.zeros((), device=self._total.device, dtype=self._total.dtype)
        self._total = self._total + w * loss
        self._names.append(str(name))
        self._weights.append(w)
        self._raws.append(loss.detach())

    # ── 读取 ──
    @property
    def total(self) -> torch.Tensor:
        """当前累加结果 ✓（仍是**图上的**张量 ✓ 可 backward ✓）。"""
        return self._total

    @property
    def detach_total(self) -> torch.Tensor:
        """detached 结果 ✓（供日志/赋值 ✓ 与旧代码 `loss = loss.detach() + …` 一致 ✓）。"""
        return self._total.detach()

    def names(self) -> List[str]:
        return list(self._names)

    def terms(self) -> List[Dict[str, object]]:
        """逐项 `{name, weight, raw}` ✓ —— `raw` 是 **detached tensor** ✓（**不**同步 ✗）。"""
        return [{"name": n, "weight": w, "raw": r}
                for n, w, r in zip(self._names, self._weights, self._raws)]

    def as_floats(self) -> Dict[str, float]:
        """★ **唯一**会做 CUDA→CPU 转换的地方 ✓（只在日志周期调用 ✓ 审查 P1-6 ✓）。"""
        return {n: float(r) for n, r in zip(self._names, self._raws)}

    def weighted_floats(self) -> Dict[str, float]:
        """带权重的数值 ✓（日志用 ✓ 同样只在日志周期调用 ✓）。"""
        return {n: float(w * r) for n, w, r in zip(self._names, self._weights, self._raws)}

    def active(self, eps: float = 0.0) -> List[str]:
        """权重非零的项名 ✓（用于"本步启用了哪些损失"的日志/清单 ✓）。"""
        return [n for n, w in zip(self._names, self._weights) if abs(w) > eps]

    def describe(self) -> str:
        """一行摘要 ✓（**会**同步 ✓ ⇒ 只在日志周期调用 ✓）。"""
        parts = [f"{n}={float(w * r):.4g}(w={w:g})"
                 for n, w, r in zip(self._names, self._weights, self._raws)]
        return " | ".join(parts)


def compose(base: torch.Tensor, terms, base_name: str = "fm") -> ObjectiveComposer:
    """便捷构造 ✓: `terms` = 可迭代的 `(name, loss, weight)` ✓（**顺序即累加顺序** ✓）。"""
    obj = ObjectiveComposer(base, base_name=base_name)
    for name, loss, weight in terms:
        obj.add(name, loss, weight)
    return obj
