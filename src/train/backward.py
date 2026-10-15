# -*- coding: utf-8 -*-
"""backward.py — **训练反向传播的三种明确策略**（2026-10-14 [重构 PR4] ✓）

审查 §4.2 原文:
  "你们当前 `strong_style.py` 在 FM backward 后再做额外 forward/backward，目的是降低
   24GB 显存峰值。这个优化应保留，但要**抽象为明确的 `sequential_backward` 策略**，
   而不是让 Critic Loss 私自掌握训练步骤。" ✓
  "建议至少支持:
     · `joint_backward`：一次 forward、一次组合 Loss backward。
     · `sequential_backward`：先释放 FM graph，再对辅助 Loss 重新 forward/backward。
     · `gradient_probe`：仅用于诊断，不参与正常训练路径。
   并且分别测量速度与显存。" ✓

本模块把这些策略**命名**并集中到一处 ✓ —— 在此之前，它们是散落在 2900 行 `main()`
里的裸 `loss.backward()` 调用 ✓（读者无法从调用点判断走的是哪条策略 ✗）。

★ 本模块是**纯封装** ✓: 每个函数体都是对既有调用的逐字转发 ✓
  ⇒ 梯度**逐位不变** ✓（见 tests/test_refactor_pr4b.py 的等价性测试 ✓）
  没有任何隐式的 zero_grad / clip / step ✗ —— 那些仍由调用方（训练循环）掌握 ✓
  （审查原则 B: "实验 Loss 不直接调用 optimizer" ✓）

三种策略的**显存/速度权衡**（供 PR6 实测 ✓）:
  · joint      : 峰值显存最高（FM 与辅助 loss 的图同时存活 ✓），但只需一次 backward ✓
  · sequential : 峰值显存最低（FM 图在第一次 backward 后即释放 ✓），代价是辅助分支
                 需要**重新 forward** ✓ ⇒ 速度略慢 ✓
  · probe      : 只在前若干步/周期执行 ✓ 用 `autograd.grad(retain_graph=True)` 取
                 梯度范数做**诊断** ✓ 不产生参数更新 ✓ 不进正常训练路径 ✓
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

__all__ = ["joint_backward", "sequential_backward", "aux_backward", "grad_norm",
           "resolve_strategy", "describe", "param_groups_for_probe", "STRATEGIES"]

STRATEGIES = ("joint", "sequential", "probe")


def joint_backward(loss) -> None:
    """**joint_backward** ✓: 一次组合 loss 的 backward ✓（等价于裸 `loss.backward()` ✓）。

    用于: 无强风格后训练（或强风格关闭时）的主路径 ✓。
    ⚠ 不调用 `optimizer.step()` / 不 zero_grad ✗ —— 生命周期归调用方 ✓。
    """
    loss.backward()


def sequential_backward(loss, aux_factory) -> bool:
    """**sequential_backward** ✓: 先 backward 主 loss（**释放 FM/REPA 激活** ✓），
    **再调用 `aux_factory()` 构造辅助 loss** 并 backward ✓。

    ★ **顺序本身就是语义** ✗ —— 提前构造辅助 loss 会让两张图同时存活 ✗
      ⇒ 完全失去"降低 24GB 峰值显存"的意义 ✗
      ⇒ 所以 API 用 `aux_factory`（**回调** ✓）而不是先算好的 `aux_loss` ✓
        —— 让"先释放、后构造"成为**类型层面无法写错**的约束 ✓✓
      （这正是审查说的"不要让 Critic Loss 私自掌握训练步骤" ✓：顺序由本模块定义 ✓）

    参数:
      loss        : 主（FM/REPA）组合 loss ✓ —— **先** backward ✓
      aux_factory : 无参回调 ✓ 在主 backward **之后**调用 ✓ 返回辅助 loss 或 None ✓
    返回: 辅助 loss 是否真的执行了 backward ✓（`requires_grad` 为假 ⇒ 跳过 ✓
          与旧实现 `if _aux_style.requires_grad: _aux_style.backward()` 逐字一致 ✓）

    ⚠ 两次 backward **累加**到同一组 `.grad` ✓ ⇒ 只需一次 clip/step ✓
      ⇒ 与 joint 的**梯度之和相同** ✓（旧注释原话:
         "Same summed gradients, one clip/optimizer step, but release FM/REPA saved
          activations before building the full style graph." ✓）
    """
    loss.backward()
    aux = aux_factory() if aux_factory is not None else None
    return aux_backward(aux)


def aux_backward(aux_loss) -> bool:
    """`sequential_backward` 的**第二半** ✓ —— 供"辅助 loss 在调用点内联构造"的场景 ✓。

    与旧实现逐字一致: `if aux.requires_grad: aux.backward()` ✓
    ⚠ 调用方必须保证主 loss **已经** backward 过 ✓（否则不是 sequential 语义 ✗）
    """
    if aux_loss is not None and bool(getattr(aux_loss, "requires_grad", False)):
        aux_loss.backward()
        return True
    return False


def grad_norm(loss, params: Iterable[torch.nn.Parameter],
              retain_graph: bool = True) -> Optional[torch.Tensor]:
    """**gradient_probe** 的原语 ✓: 对给定参数求 `loss` 的梯度**范数** ✓（不累积到 .grad ✓）。

    审查 §7.3: "强风格梯度探针按周期和代表层执行 —— 全模型额外求导可能较昂贵" ✓
      ⇒ 本函数只负责"求一次范数" ✓；**何时探、探哪些参数**由调用方决定 ✓
        （周期 ✓ 代表层 ✓ 都属调用方的策略 ✓ 不写死在本模块 ✗）

    返回: 标量范数 ✓；全部梯度为 None ⇒ 返回 None ✓（不伪造 0 ✓）
    注意: `retain_graph=True` ✓ 是因为探针**不消耗**计算图（后续还要 backward ✓）
    """
    ps = [p for p in params if p.requires_grad]
    if not ps:
        return None
    grads = torch.autograd.grad(loss, ps, retain_graph=retain_graph, allow_unused=True)
    acc = None
    for g in grads:
        if g is None:
            continue
        s = (g ** 2).sum()
        acc = s if acc is None else acc + s
    if acc is None:
        return None
    return torch.sqrt(acc)


def resolve_strategy(args) -> str:
    """从**真实配置**解析当前 run 走的是哪条策略 ✓（供日志/manifest 记录 ✓ P1-7 ✓）。

    规则（按优先级 ✓ 与训练循环的实际分支一致 ✓）:
      · 强风格后训练开启（`strong_style_mode != 'off'`）⇒ **sequential** ✓
        （它的辅助 loss 在主 backward 之后才构造 ✓ ⇒ 天然是 sequential ✓）
      · 否则 ⇒ **joint** ✓
    `probe` 不是互斥策略 ✓ —— 它是**周期性诊断** ✓ 是否开启见 `describe()` ✓
    """
    mode = str(getattr(args, "strong_style_mode", "off") or "off")
    if mode != "off":
        return "sequential"
    return "joint"


def describe(args) -> str:
    """一行策略摘要 ✓（写入日志 ✓ 便于事后按策略分组对比吞吐/显存 ✓）。"""
    strat = resolve_strategy(args)
    bits = [f"backward={strat}"]
    if strat == "sequential":
        bits.append(f"strong_style_mode={getattr(args, 'strong_style_mode', '?')}")
        bits.append(f"cf_every={getattr(args, 'strong_style_cf_every', '?')}")
    _pe = int(getattr(args, "strong_style_probe_every", 0) or 0)
    _ps = int(getattr(args, "probe_style_grad_steps", 0) or 0)
    if _pe or _ps:
        bits.append(f"probe(every={_pe or '-'}, first={_ps or '-'})")
    else:
        bits.append("probe=off")
    return " ".join(bits)


def param_groups_for_probe(model: torch.nn.Module,
                           layers: Optional[Sequence[int]] = None) -> list:
    """探针用的参数子集 ✓（`layers=None` ⇒ 全部可训练参数 ✓）。

    审查建议"按周期和**代表层**执行" ✓ —— 本函数提供"取代表层参数"的**手段** ✓，
    是否使用由调用方决定 ✓（默认保持现状 = 全模型 ✓ 行为不变 ✓）。
    """
    ps = [p for p in model.parameters() if p.requires_grad]
    if not layers:
        return ps
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for li in layers:
            if f"blocks.{li}." in name:
                out.append(p)
                break
    return out or ps
