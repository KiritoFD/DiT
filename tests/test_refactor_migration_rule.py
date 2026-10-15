# -*- coding: utf-8 -*-
"""test_refactor_migration_rule.py —— **PR5 调用点迁移规则**（2026-10-14 ✓）

★ 背景: 审查 §4.1 要求把散落的 `g`/`y_char`/`y_callig`/`y_pair`/`y_script` 收进
  `ConditionBatch` ✓，并且"迁移可以保留旧导入路径作为**短期兼容层**" ✓。

★★ 迁移中有一个**已实测确认**的陷阱 ✗✗:
  旧调用点（如 `src/eval/sampler.py`）的写法是"**存在才加键**" ✓::

      mk = dict(y_callig=yc, y_char=yh)
      if skel is not None:
          mk["g"] = skel[i:j]              # ← 不存在就**根本不传** g ✓

  而 `batch_to_kwargs(b, for_forward=True)` **总是保留** `g` ✓（哪怕 None ✓）
    —— 契约上 `g=None` 是"**条件被丢掉**"的语义 ✓ 而不是"不传" ✓
    ⇒ 直接替换会把"没传 g"变成"明确传 g=None" ✗ ⇒ **语义改变** ✗

  实测::

      当前构造键 = ['y_callig', 'y_char']
      迁移后键   = ['g', 'y_callig', 'y_char']      ← 凭空多了 g ✗
      等价?      = False

  ⇒ 本文件锁定 `kwargs_for_call()` 这个**安全迁移助手** ✓

★ 另外记录: `cond` 键（ControlNet 包装用 ✓）**不在** ConditionBatch 里 ✗
  ⇒ 那部分必须**留在调用方** ✓（不要硬塞进兼容层 ✗）
"""
from __future__ import annotations

import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.conditioning import (batch_to_kwargs, build_batch,        # noqa: E402
                                    kwargs_for_call)


def _t(n=2):
    return torch.zeros(n, dtype=torch.long)


# ─────────────── ★ 陷阱本身（实测值 ✓）───────────────
def test_for_forward_adds_g_that_was_never_passed():
    """★★ 复现陷阱 ✓: `for_forward=True` 会**凭空**加上 `g` ✗（哪怕从没传过 ✓）"""
    b = build_batch(y_callig=_t(), y_char=_t())
    old_style = {"y_callig": _t(), "y_char": _t()}          # 旧代码"存在才加" ✓
    naive = batch_to_kwargs(b, for_forward=True)
    assert sorted(old_style) == ["y_callig", "y_char"]
    assert "g" in naive and naive["g"] is None, \
        "`for_forward=True` 应**总是**带 g（哪怕 None ✓）—— 这正是陷阱 ✓"
    assert sorted(naive) != sorted(old_style), "两者**不等价** ✗（实测确认 ✓）"


# ─────────────── ★ 安全助手: 与旧写法逐字等价 ───────────────
def test_kwargs_for_call_matches_old_style_exactly():
    """★★ `kwargs_for_call` 必须与"存在才加键"的旧写法**逐字等价** ✓"""
    cases = [
        dict(),                                                     # 只有 y_callig/y_char ✓
        dict(hier=True),                                            # + 三个 S2 标签 ✓
        dict(skel=True),                                            # + g ✓
        dict(hier=True, skel=True),                                 # 全有 ✓
    ]
    for extra in cases:
        # 旧写法（逐字复刻 ✓）
        old = {"y_callig": _t(), "y_char": _t()}
        if extra.get("hier"):
            old.update(y_callig_raw=_t(), y_pair=_t(), y_script=_t())
        if extra.get("skel"):
            old["g"] = torch.zeros(2, 1, 4, 4)
        # 新写法 ✓
        b = build_batch(y_callig=old["y_callig"], y_char=old["y_char"],
                        y_callig_raw=old.get("y_callig_raw"),
                        y_pair=old.get("y_pair"), y_script=old.get("y_script"),
                        g=old.get("g"))
        new = kwargs_for_call(b)
        assert sorted(new) == sorted(old), f"{extra}: 键集合不同 ✗ {sorted(new)} vs {sorted(old)}"
        for k in old:
            assert torch.equal(new[k], old[k]), f"{extra}: {k} 值不同 ✗"


def test_kwargs_for_call_drops_all_none_by_default():
    b = build_batch(y_callig=_t(), y_char=_t())
    assert set(kwargs_for_call(b)) == {"y_callig", "y_char"}


def test_kwargs_for_call_can_explicitly_drop_glyph():
    """★ 只有**确实**要表达"明确不带字形"时才 `allow_none_glyph=True` ✓"""
    b = build_batch(y_callig=_t(), y_char=_t())
    kw = kwargs_for_call(b, allow_none_glyph=True)
    assert "g" in kw and kw["g"] is None, "显式表达时应带 g=None ✓"
    assert set(kw) == {"y_callig", "y_char", "g"}


def test_kwargs_for_call_keeps_real_glyph():
    b = build_batch(y_callig=_t(), y_char=_t(), g=torch.ones(2, 1, 4, 4))
    kw = kwargs_for_call(b)
    assert "g" in kw and torch.equal(kw["g"], torch.ones(2, 1, 4, 4))


# ─────────────── ★ 边界: cond 键不归本层管 ───────────────
def test_cond_key_is_not_part_of_condition_batch():
    """★ `cond`（ControlNet 包装 ✓）**不在**兼容层里 ✗ ⇒ 必须留在调用方 ✓"""
    b = build_batch(y_callig=_t())
    for fn in (kwargs_for_call, lambda x: batch_to_kwargs(x, for_forward=False)):
        assert "cond" not in fn(b), "`cond` 不应由 ConditionBatch 产生 ✗（那是包装器的键 ✓）"


def test_sampler_still_uses_existence_based_construction():
    """★ 特性化守卫 ✓: 记录 `sampler.py` 目前仍是"存在才加键"的写法 ✓

    若将来迁移它 ⇒ 必须改用 `kwargs_for_call` ✓ 或显式保留"存在才加"语义 ✓
    ⇒ 本测试会在**迁移未配套**时失败 ✓（提醒去核对 ✓）
    """
    src = io.open("src/eval/sampler.py", encoding="utf-8").read()
    assert "mk = dict(y_callig=yc, y_char=yh)" in src, \
        "sampler 的构造方式变了 ✗ ⇒ 请核对迁移是否保持了'存在才加键'语义 ✓"
    assert "if hier_conds is not None:" in src and "if skel is not None:" in src, \
        "sampler 的按存在加键结构变了 ✗ ⇒ 请核对语义 ✓"
    # 且**尚未**迁移到 kwargs_for_call ✓（迁移后本断言需一并更新 ✓）
    assert "kwargs_for_call" not in src, \
        "sampler 已迁移 ✓ ⇒ 请把本守卫改成断言它**用了** kwargs_for_call ✓"
