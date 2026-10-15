# -*- coding: utf-8 -*-
"""test_refactor_pr4g.py —— 2026-10-14 [重构 PR4] **NonFiniteGuard** 的验收测试 ✓

审查 §4.2 第 5 条: "finite 梯度检查和 clipping" ✓
审查原文（旧注释）: "NaN guard: skip clip/step/ema/accumulate if loss is not finite
  (e.g. a bad sample)." ✓
  "已 backward -> `.grad` 里留下 NaN，但**不** clip/step/ema -> 参数不被污染；
   这些 NaN 梯度由下一轮顶部的 `zero_grad(set_to_none=True)` 释放。" ✓

本组测试锁定:
  ① `check` ≡ `bool(torch.isfinite(loss))` ✓（含 nan / inf / 正常 ✓）
  ② `note_skipped` 计数 ✓ + **仅 rank0** 告警 ✓ + **告警文本逐字一致** ✓
  ③ ★ `float(loss_diff)` **只在出问题时**求值 ⇒ 不在热路径 ✓
  ④ 边界 ✓: 它**不**撤销 backward ✗ 也**不**清梯度 ✗（那是 zero_grad 的事 ✓）
"""
from __future__ import annotations

import ast
import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.step import NonFiniteGuard                                  # noqa: E402


class _Log:
    def __init__(self):
        self.msgs = []

    def warning(self, m, *a):
        self.msgs.append(m % a if a else m)

    def info(self, m, *a):
        pass


# ─────────────── ① check ≡ 旧表达式 ───────────────
def test_check_matches_torch_isfinite():
    g = NonFiniteGuard()
    for v in (0.0, 1.0, -3.5, 1e30):
        assert g.check(torch.tensor(v)) is bool(torch.isfinite(torch.tensor(v)))
    for bad in (float("nan"), float("inf"), float("-inf")):
        t = torch.tensor(bad)
        assert g.check(t) is False, f"{bad} 必须判为**非有限** ✗"
        assert g.check(t) is bool(torch.isfinite(t)), "必须与旧表达式逐字一致 ✓"


def test_check_on_multielement_tensor_is_all():
    """旧代码对 0 维 loss 用 `bool(torch.isfinite(loss))` ✓；
    多元素张量时 `bool()` 会报错 ✓ —— 这里记录**同一行为** ✓（不得悄悄改成 all() ✗）"""
    g = NonFiniteGuard()
    try:
        g.check(torch.tensor([1.0, 2.0]))
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "多元素张量应与旧代码一样报错 ✓（不得静默改成 all() ✗）"


# ─────────────── ② 计数 + 告警 ───────────────
def test_note_skipped_counts_and_warns_on_rank0():
    lg = _Log()
    g = NonFiniteGuard(lg, rank=0)
    g.note_skipped(5000, torch.tensor(0.9876))
    assert g.nan_steps == 1
    assert len(lg.msgs) == 1
    assert lg.msgs[0] == ("Step 5000 skipped: non-finite loss "
                          "(diff=0.9876). Accumulated skips: 1"), lg.msgs[0]


def test_warning_text_matches_old_format():
    """★ 告警文本必须与旧实现**逐字一致** ✓（便于按日志检索历史 ✓）"""
    lg = _Log()
    g = NonFiniteGuard(lg, rank=0)
    g.note_skipped(12345, torch.tensor(1.5))
    old = ("Step 12345 skipped: non-finite loss "
           f"(diff={float(torch.tensor(1.5)):.4f}). Accumulated skips: {1}")
    assert lg.msgs[0] == old


def test_rank_nonzero_does_not_warn_but_still_counts():
    lg = _Log()
    g = NonFiniteGuard(lg, rank=1)
    g.note_skipped(10, torch.tensor(0.1))
    g.note_skipped(11, torch.tensor(0.2))
    assert g.nan_steps == 2, "计数与 rank 无关 ✓"
    assert lg.msgs == [], "非 rank0 不得打日志 ✓"


def test_no_logger_is_safe():
    g = NonFiniteGuard(None, rank=0)
    g.note_skipped(1, torch.tensor(0.0))
    assert g.nan_steps == 1


def test_accumulates_across_steps():
    g = NonFiniteGuard(_Log(), rank=0)
    for i in range(5):
        g.note_skipped(i, torch.tensor(float(i)))
    assert g.nan_steps == 5
    assert "Accumulated skips: 5" in g.logger.msgs[-1]


# ─────────────── ③ 不在热路径 ───────────────
def test_loss_diff_not_evaluated_when_none():
    """★ `float(loss_diff)` 只在**出问题时**才求值 ✓ ⇒ 不在热路径 ✓"""
    lg = _Log()
    g = NonFiniteGuard(lg, rank=0)
    g.note_skipped(9, None)                     # 不传 loss_diff ⇒ 不得崩 ✓
    assert g.nan_steps == 1
    assert "diff=" not in lg.msgs[0], "无 loss_diff 时不得输出 diff 段 ✓"


def test_loss_diff_evaluated_lazily():
    """用一个**会爆炸**的对象证明: 只有真正取 float 时才求值 ✓"""
    class _Boom:
        def __float__(self):
            raise AssertionError("不得被求值 ✗")
    g = NonFiniteGuard(_Log(), rank=1)          # rank1 ⇒ 直接 return ✓ 不取 float ✓
    g.note_skipped(1, _Boom())                  # 不得触发 __float__ ✓
    assert g.nan_steps == 1


# ─────────────── ④ 边界（不撤销 backward / 不清梯度 ✓）───────────────
def test_guard_does_not_touch_gradients():
    """★ 本模块**只**回答"要不要 step" ✓ —— 不撤销 backward ✗ 不清梯度 ✗"""
    src = io.open("src/train/step.py", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    for bad in ("zero_grad", "backward", "clip_grad", "opt.step", "state_dict"):
        assert bad not in code, f"step.py 不得涉及 `{bad}` ✗（职责边界 ✓）"


def test_stats_and_describe():
    g = NonFiniteGuard(_Log(), rank=0)
    g.note_skipped(1, torch.tensor(0.0))
    assert g.stats() == {"nan_steps": 1}
    assert g.describe() == "non_finite_skips=1"


# ─────────────── 接线守卫 ───────────────
def test_train_py_uses_guard():
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "_guard = _NonFiniteGuard(logger, rank)" in t, "未构造守卫 ✗"
    assert "_finite = _guard.check(loss)" in t, "判定未接线 ✗"
    assert "_guard.note_skipped(train_steps, loss_diff)" in t, "跳过分支未接线 ✗"
    assert "nan_steps" not in t, "旧的 `nan_steps` 标量必须已全部迁移 ✗"
    # 旧的告警文本不得再出现在 train.py ✓（已移入组件 ✓）
    assert "Accumulated skips:" not in t, "告警文本应只在组件里 ✓"
