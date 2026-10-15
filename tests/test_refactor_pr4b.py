# -*- coding: utf-8 -*-
"""test_refactor_pr4b.py —— 2026-10-14 [重构 PR4] **backward 策略模块** 的验收测试 ✓

审查 §4.2: "建议至少支持: `joint_backward` / `sequential_backward` / `gradient_probe`" ✓
  "并且分别测量速度与显存" ✓
  "不要…让 Critic Loss 私自掌握训练步骤" ✓

本组测试锁定:
  ① 三个策略的**梯度逐位等价** ✓（封装不得改变数学 ✓）
  ② ★ **顺序不变量** ✓: sequential 必须"先主 backward（释放图）再构造辅助 loss" ✓
     —— 这是"省 24GB 峰值显存"的**唯一来源** ✓ 也是本模块存在的理由 ✓
  ③ 策略解析/描述 ✓ 与 train.py 的**接线守卫** ✓（不得再有裸 `loss.backward()` ✗）
"""
from __future__ import annotations

import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import backward as BW                                       # noqa: E402


def _mk_model(seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(),
                            torch.nn.Linear(8, 4))
    x = torch.randn(5, 6)
    return m, x


def _grads(m):
    return [None if p.grad is None else p.grad.detach().clone() for p in m.parameters()]


# ─────────────── ① 梯度逐位等价 ───────────────
def test_joint_backward_identical_to_raw():
    m, x = _mk_model(1)
    loss = m(x).pow(2).mean()
    loss.backward()
    ref = _grads(m)

    m2, x2 = _mk_model(1)
    m2.load_state_dict(m.state_dict())
    loss2 = m2(x2).pow(2).mean()
    BW.joint_backward(loss2)
    got = _grads(m2)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b), "joint_backward 必须与裸 loss.backward() 逐位一致 ✗"


def test_aux_backward_matches_conditional_call():
    for requires in (True, False):
        m, x = _mk_model(2)
        y = m(x)
        aux = y.pow(2).mean()
        if not requires:
            aux = aux.detach()
        did = BW.aux_backward(aux)
        assert did is requires, (did, requires)
        assert (m[0].weight.grad is not None) is requires
    assert BW.aux_backward(None) is False, "None 必须安全跳过 ✓"


def test_sequential_backward_sums_grads_like_joint():
    """★ sequential 与 joint 的**梯度之和必须相同** ✓（审查原话: "Same summed gradients" ✓）"""
    m, x = _mk_model(3)
    main = m(x).pow(2).mean()
    # joint: 把辅助项也算进同一个 loss 再一次 backward ✓
    aux_val = (m(x) * 0.5).pow(2).mean()
    (main + aux_val).backward()
    ref = _grads(m)

    m2, x2 = _mk_model(3)
    m2.load_state_dict(m.state_dict())
    main2 = m2(x2).pow(2).mean()
    BW.sequential_backward(main2, lambda: (m2(x2) * 0.5).pow(2).mean())
    got = _grads(m2)
    for a, b in zip(ref, got):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6), "梯度之和必须一致 ✓"


# ─────────────── ② ★ 顺序不变量（本模块存在的理由 ✓）───────────────
def test_sequential_calls_factory_after_main_backward():
    """★★ 核心不变量 ✓: `aux_factory` 必须在**主 backward 之后**才被调用 ✓

    若提前调用 ⇒ 两张图同时存活 ✗ ⇒ 完全失去"降低峰值显存"的意义 ✗
    ⇒ 本模块用**回调**把顺序变成类型层面无法写错的约束 ✓

    ⚠ 探针必须**无歧义** ✗ —— 早期版本用 `loss.grad_fn is None` 判断"已反传" ✗
      但 backward **之后 `grad_fn` 仍非 None** ✗（只有 saved tensors 被释放 ✓）
      ⇒ 那是**假阴性** ✓（实测得到 `main_not_backwarded` ✗）
      ⇒ 改为在**两侧**用自定义 `autograd.Function` 直接打点 ✓ 顺序一目了然 ✓
    """
    m, x = _mk_model(4)
    events = []

    class _Main(torch.autograd.Function):
        """主 loss 的反向会在真正的 backward 阶段打点 ✓"""
        @staticmethod
        def forward(ctx, t):
            return t

        @staticmethod
        def backward(ctx, g):
            events.append("main_backward")
            return g

    class _Aux(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t

        @staticmethod
        def backward(ctx, g):
            events.append("aux_backward")
            return g

    main = _Main.apply(m(x).pow(2).mean())

    def _factory():
        events.append("factory_called")
        return _Aux.apply(m(x).pow(2).mean())

    BW.sequential_backward(main, _factory)
    assert events == ["main_backward", "factory_called", "aux_backward"], \
        f"sequential 顺序必须是 主反传 -> 构造辅助 -> 辅助反传 ✗ 实际 {events}"


def test_sequential_skips_factory_when_none():
    m, x = _mk_model(5)
    main = m(x).pow(2).mean()
    assert BW.sequential_backward(main, None) is False


# ─────────────── ③ 探针原语 ───────────────
def test_grad_norm_matches_inline_formula():
    m, x = _mk_model(6)
    loss = m(x).pow(2).mean()
    ps = [p for p in m.parameters() if p.requires_grad]
    ref_g = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    ref = torch.sqrt(sum((g ** 2).sum() for g in ref_g if g is not None))
    got = BW.grad_norm(loss, ps)
    assert torch.equal(ref, got), "grad_norm 必须与旧内联公式逐位一致 ✗"
    # 探针不得累积到 .grad ✓
    assert all(p.grad is None for p in m.parameters()), "探针不得写入 .grad ✗"


def test_grad_norm_returns_none_without_params():
    m, x = _mk_model(7)
    loss = m(x).pow(2).mean()
    assert BW.grad_norm(loss, []) is None, "无参数 ⇒ None ✓（不伪造 0 ✗）"


def test_param_groups_for_probe_layers_and_fallback():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    m = _M()
    allp = BW.param_groups_for_probe(m)
    assert len(allp) == len([p for p in m.parameters() if p.requires_grad])
    only1 = BW.param_groups_for_probe(m, layers=[1])
    assert len(only1) == 2, only1                      # blocks.1 的 weight + bias ✓
    assert len(BW.param_groups_for_probe(m, layers=[99])) == len(allp), \
        "无命中 ⇒ 回退到全部 ✓（不得返回空 ✗）"


# ─────────────── ④ 策略解析 + 接线守卫 ───────────────
def test_resolve_and_describe():
    class A:
        strong_style_mode = "off"
    class B:
        strong_style_mode = "D"
        strong_style_cf_every = 4
        strong_style_probe_every = 400
        probe_style_grad_steps = 0
    assert BW.resolve_strategy(A()) == "joint"
    assert BW.resolve_strategy(B()) == "sequential"
    da, db = BW.describe(A()), BW.describe(B())
    assert "backward=joint" in da and "probe=off" in da, da
    assert "backward=sequential" in db and "probe(every=400" in db, db
    assert set(BW.STRATEGIES) == {"joint", "sequential", "probe"}


def test_train_py_has_no_bare_backward_and_uses_module():
    """★ 接线守卫 ✓: train.py 不得再有**裸** `loss.backward()` ✗ 必须走命名策略 ✓"""
    import re
    s = io.open("src/train/train.py", encoding="utf-8").read()
    assert not re.search(r"^\s+loss\.backward\(\)", s, re.M), \
        "train.py 仍有裸 loss.backward() ✗（PR4: 必须走命名策略 ✓）"
    assert not re.search(r"^\s+_aux_style\.backward\(\)", s, re.M), \
        "train.py 仍有裸 _aux_style.backward() ✗（应走 aux_backward ✓）"
    assert "_bw.joint_backward(" in s and "_bw.aux_backward(" in s, "未接线到策略模块 ✗"
    assert s.count("_bw.grad_norm(") >= 5, "探针未全部改走 grad_norm ✗"
    assert "[backward]" in s, "启动日志必须打印策略摘要 ✓（审查 §4.2 要求可分组测量 ✓）"
